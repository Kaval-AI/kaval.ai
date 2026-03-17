import asyncio
import logging
import json
from typing import Type, Optional

from pydantic import BaseModel, Field, ConfigDict

from kavalai.agents.run_context import RunContext
from kavalai.functionkernel import FunctionKernel
from kavalai.llm_clients.llm_client import LLMClient
from kavalai.llm_clients.common import Streamer

logger = logging.getLogger(__name__)


class ToolCall(BaseModel):
    """This data structure represents tool call requests."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Tool call name i.e python://mypackage.myfunc")
    call_id: Optional[str] = Field(
        description="Generate an ID, which represents this result in downstream agent runs."
    )
    args: str = Field(
        default="{}",
        description="Arguments for the tool call in JSON format.",
    )
    persist_to: Optional[str] = Field(
        default=None, description="Key to store in run_context.data"
    )


def get_step_output_type(ResponseModel=Type[BaseModel]):
    class StepOutput(BaseModel):
        """Data structure that helps passing around information between consecutive agent runs."""

        model_config = ConfigDict(extra="forbid")

        short_explanation: str = Field(
            description="Human friendly summary of planned steps. Be very concise, maximum 50 characters.",
            max_length=50,
        )
        long_explanation: str = Field(
            description="Add exaplanations or instructions for downstream LLM-calls.",
            max_length=500,
        )
        step_number: int = Field(description="Current iteration of planning agent.")
        max_steps: int = Field(
            description="If step_number == max_steps - 1, then you must output something, because agent will be terminated."
        )
        tool_calls: list[ToolCall] = Field(
            description="Add tool call requests here, their output will be stored in planner_context accessible with `call_id` key."
        )
        output: Optional[ResponseModel] = Field(
            description="Final output. If tool calls with call_id are used, you can leave fields with matching names empty in this model, and they will be automatically populated from planner_context."
        )

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
        self._streamer = streamer
        self._temperature = temperature
        self._stream_updates = stream_updates
        self._stream_output = stream_output
        self._planner_context = {}
        self._step_outputs = []

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
                "Each tool call MUST be a valid JSON object matching the ToolCall structure with these fields:",
                "- name (REQUIRED): The tool URI, e.g., 'python://mypackage.myfunc'",
                "- args (REQUIRED): A JSON string containing all required arguments for the tool",
                "- call_id (OPTIONAL): An identifier to reference this tool's result in LATER STEPS within planner_context. Use this when you need the result in subsequent planning iterations.",
                "- persist_to (OPTIONAL): A key to permanently store the result in run_context.data. Use this when the result needs to be available OUTSIDE the planning agent (e.g., for other agents or final output).",
                "",
                "IMPORTANT: Use call_id for intermediate results needed in later planning steps. Use persist_to for results that should persist beyond planning.",
                "",
                "Examples:",
                '1. Basic tool call: {"name": "python://mypackage.myfunc", "args": "{\\"param1\\": \\"value1\\", \\"param2\\": 10}"}',
                '2. With call_id (for reuse in later steps): {"name": "python://data.fetch", "call_id": "user_data", "args": "{\\"user_id\\": 123}"}',
                '3. With persist_to (for final persistence): {"name": "python://report.generate", "persist_to": "final_report", "args": "{\\"format\\": \\"pdf\\"}"}',
                '4. With both (reuse AND persist): {"name": "python://analysis.run", "call_id": "analysis_result", "persist_to": "analysis_output", "args": "{}"}',
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
                f"max_steps={max_iterations}",
                "Step Outputs (previous steps):",
                json.dumps(
                    [
                        so.model_dump() if hasattr(so, "model_dump") else so
                        for so in self._step_outputs
                    ],
                    indent=2,
                ),
            ]
            system_prompt = "\n\n".join(prompt_parts)

            logger.info(f"Running iteration {iter_no} for task: {task}")

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

            logger.info(f"Step {iter_no}: {step_output.short_explanation}")

            if self._streamer and self._stream_updates:
                await self._streamer.stream_complete(
                    step_output.short_explanation, name="running_task"
                )

            if step_output.tool_calls:

                async def _call_tool(tool_call):
                    try:
                        args = (
                            json.loads(tool_call.args)
                            if isinstance(tool_call.args, str)
                            else tool_call.args
                        )
                    except Exception as e:
                        logger.error(f"Failed to parse tool args: {e}")
                        args = {}

                    logger.info(f"Calling tool {tool_call.name} with {args}")
                    try:
                        tool_result = await self._kernel.call_tool(
                            tool_uri=tool_call.name,
                            arguments=args,
                        )
                    except Exception as e:
                        logger.error(f"Tool {tool_call.name} failed: {e}")
                        tool_result = f"Error: {e}"

                    return tool_call, args, tool_result

                results = await asyncio.gather(
                    *[_call_tool(tc) for tc in step_output.tool_calls]
                )

                for tool_call, args, tool_result in results:
                    if tool_call.call_id:
                        self._planner_context[tool_call.call_id] = tool_result

                    if tool_call.persist_to:
                        self._run_context.data[tool_call.persist_to] = tool_result

                    agent_service = getattr(self._run_context, "agent_service", None)
                    if agent_service and hasattr(agent_service, "add_task"):
                        try:
                            await agent_service.add_task(
                                agent_id=self._run_context.agent_id,
                                session_id=self._run_context.session_id,
                                run_id=self._run_context.run_id,
                                name=tool_call.name,
                                inputs={"arguments": args},
                                output=tool_result
                                if not hasattr(tool_result, "model_dump")
                                else tool_result.model_dump(),
                            )
                        except TypeError:
                            pass

            if step_output.output is not None:
                final_output = step_output.output
                if not step_output.tool_calls:
                    break

        if final_output is not None:
            if self._streamer and self._stream_output:
                await self._streamer.stream_complete(final_output.model_dump_json())
            return final_output

        return None
