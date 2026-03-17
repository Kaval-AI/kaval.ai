import asyncio
from loguru import logger
import json
import time
from typing import Type, Optional

from pydantic import BaseModel, Field, ConfigDict

from kavalai.agents.run_context import RunContext
from kavalai.agents.agent_service import AgentService
from kavalai.functionkernel import FunctionKernel
from kavalai.llm_clients.llm_client import LLMClient
from kavalai.llm_clients.common import Streamer


class ToolCall(BaseModel):
    """This data structure represents tool call requests."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="Tool call name i.e python://websearch.serper_web_search"
    )
    args: str = Field(
        default="{}",
        description="A JSON string of arguments for the tool call. Use template strings like {{context.key}} or {{input.key}} to reference data without copying it.",
    )
    call_id: Optional[str] = Field(
        default=None,
        description="Generate an ID, which represents this result in downstream agent runs.",
    )
    persist_to: Optional[str] = Field(
        default=None, description="Key to store in run_context.data"
    )


def get_step_output_type(ResponseModel=Type[BaseModel]):
    class StepOutput(BaseModel):
        """Data structure that helps passing around information between consecutive agent runs."""

        short_explanation: str = Field(
            description="Human friendly summary of planned steps. Be very concise, maximum 50 characters.",
            max_length=50,
        )
        long_explanation: str = Field(
            description="Add exaplanations or instructions for downstream LLM-calls.",
            max_length=500,
        )
        tool_calls: list[ToolCall] = Field(
            default=[],
            description="Add tool call requests here, their output will be stored in planner_context accessible with `call_id` key.",
        )
        output: Optional[ResponseModel] = None

    return StepOutput


class PlanningAgent:
    def __init__(
        self,
        *,
        kernel: FunctionKernel,
        run_context: RunContext,
        llm_client: LLMClient,
        input_data: dict[str, dict],
        response_model: Type[BaseModel] = BaseModel,
        agent_service: Optional[AgentService] = None,
        streamer: Optional[Streamer] = None,
        temperature: Optional[float] = None,
        stream_updates: bool = False,
        stream_output: bool = False,
    ):
        self._kernel = kernel
        self._run_context = run_context
        self._llm_client = llm_client
        self._input_data = input_data
        self._response_model = response_model
        self._agent_service = agent_service
        self._streamer = streamer
        self._temperature = temperature
        self._stream_updates = stream_updates
        self._stream_output = stream_output
        self._planner_context = {}
        self._step_outputs = []

    def _resolve_template(self, value: any) -> any:
        """Resolves template strings like {{context.key}} or {{input.key}}."""
        if isinstance(value, str):
            if value.startswith("{{context.") and value.endswith("}}"):
                key = value[10:-2].strip()
                return self._planner_context.get(key, value)
            elif value.startswith("{{input.") and value.endswith("}}"):
                key = value[8:-2].strip()
                return self._input_data.get(key, value)
        elif isinstance(value, dict):
            return {k: self._resolve_template(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._resolve_template(v) for v in value]
        return value

    async def _call_tool(
        self, tool_call: ToolCall
    ) -> tuple[ToolCall, dict, any, float]:
        """Resolves arguments and executes a single tool call."""
        duration = 0.0

        # Parse args from JSON string
        try:
            args_dict = json.loads(tool_call.args)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse args as JSON: {tool_call.args}")
            args_dict = {}

        args = self._resolve_template(args_dict)

        logger.info(f"Calling tool {tool_call.name} with {args}")
        try:
            start_time = time.perf_counter()
            tool_result = await self._kernel.call_tool(
                tool_uri=tool_call.name,
                arguments=args,
            )
            duration = time.perf_counter() - start_time
        except Exception as e:
            logger.error(f"Tool {tool_call.name} failed: {e}")
            tool_result = f"Error: {e}"

        return tool_call, args, tool_result, duration

    async def run(
        self, task: str, chat_history: list[dict] = None, max_iterations: int = 10
    ):
        if chat_history is None:
            chat_history = []

        StepOutput = get_step_output_type(self._response_model)
        final_output = None

        for iter_no in range(max_iterations):
            prompt_parts = [
                "You are a planning agent. Your goal is to achieve the following task:",
                task.strip(),
                "# Tool calling instructions:",
                "Use tools to fulfill the task."
                "To call a tool, provide its name and arguments as a JSON string. Example:",
                '{"name": "python://websearch.serper_web_search", "args": "{\\"query\\": \\"Kaval AI\\"}"}',
                "",
                "You can use template strings to reference data without copying it:",
                "- {{ context.key }}: Reference a result from a previous tool call via call_id.",
                "- {{ input.key }}: Reference data from the provided # Inputs section.",
                "",
                "Example with templates:",
                '{"name": "python://data.process", "args": "{\\"raw_data\\": \\"{{ context.fetch_result }}\\", \\"user_id\\": \\"{{ input.user_id }}\\"}", "call_id": "processed_data"}',
                "",
                "# Available tools:",
                await self._kernel.get_tool_descriptions(),
                "# Inputs:",
                json.dumps(self._input_data, indent=2),
                "Planner Context (tool results from previous steps, accessible via call_id):",
                json.dumps(
                    {
                        k: v.model_dump() if hasattr(v, "model_dump") else v
                        for k, v in self._planner_context.items()
                    },
                    indent=2,
                ),
                "",
                "PLanning data",
                f"current_step={iter_no}",
                f"max_steps={max_iterations}",
                "",
                "Step Outputs (previous steps):",
                json.dumps(
                    [
                        so.model_dump() if hasattr(so, "model_dump") else so
                        for so in self._step_outputs
                    ],
                    indent=2,
                ),
            ]
            system_prompt = "\n".join(prompt_parts)

            logger.info(f"Running iteration {iter_no} for task: {task}")
            logger.info(f"System prompt:\n{system_prompt}")

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "Analyze the situation and provide the next step output.",
                },
            ] + chat_history

            result, stats = await self._llm_client.chat_completions(
                messages=messages,
                response_model=StepOutput,
                streamer=self._streamer,
                temperature=self._temperature,
            )

            step_output: StepOutput = result
            self._step_outputs.append(step_output)

            if self._agent_service:
                try:
                    await self._agent_service.add_model_call_stats(
                        stats=stats, agent_id=self._run_context.agent_id
                    )
                except Exception as e:
                    logger.error(f"Failed to store LLM stats: {e}")

            logger.info(f"Step {iter_no}: {step_output.short_explanation}")
            logger.info(f"Tool calls {iter_no}: {step_output.tool_calls}")

            if self._streamer and self._stream_updates:
                await self._streamer.stream_complete(
                    step_output.short_explanation, name="running_task"
                )

            if step_output.tool_calls:
                results = await asyncio.gather(
                    *[self._call_tool(tc) for tc in step_output.tool_calls]
                )

                for tool_call, args, tool_result, duration in results:
                    if tool_call.call_id:
                        self._planner_context[tool_call.call_id] = tool_result

                    if tool_call.persist_to:
                        self._run_context.data[tool_call.persist_to] = tool_result

                    if self._agent_service:
                        try:
                            await self._agent_service.add_task(
                                agent_id=self._run_context.agent_id,
                                session_id=self._run_context.session_id,
                                run_id=self._run_context.run_id,
                                name=tool_call.name,
                                inputs={"arguments": args},
                                output=tool_result,
                                duration_seconds=duration,
                            )
                        except Exception as e:
                            logger.error(f"Failed to record task in agent_service: {e}")

            if step_output.output is not None:
                final_output = step_output.output
                if not step_output.tool_calls:
                    break

        if final_output is not None:
            if self._streamer and self._stream_output:
                await self._streamer.stream_complete(final_output.model_dump_json())
            return final_output

        return None
