import asyncio
import json
import time
from typing import Type, Optional

from loguru import logger
from pydantic import BaseModel, Field, ConfigDict

from kavalai.agents.agent_service import AgentService
from kavalai.agents.run_context import RunContext
from kavalai.agents.task_logger import TaskLogger
from kavalai.functionkernel import FunctionKernel
from kavalai.llm_clients.common import Streamer
from kavalai.llm_clients.llm_client import LLMClient
from kavalai.agents.workflow_model import to_plain


class ToolCall(BaseModel):
    """This data structure represents tool call requests."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="Tool call name i.e python://websearch.serper_web_search"
    )
    literal_args: str = Field(
        default="{}",
        description="Literal values to use as arguments for the tool call.",
    )
    planner_context_args: str = Field(
        default="{}",
        description="Map of tool argument names to keys in planner_context.",
    )
    input_args: str = Field(
        default="{}",
        description="Map of tool argument names to keys in input_data.",
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
        instructions: str = Field(
            description="Store very brief instructions for downstream LLM.",
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
        task_logger: Optional[TaskLogger] = None,
        streamer: Optional[Streamer] = None,
        temperature: Optional[float] = None,
        stream_updates: bool = False,
        stream_output: bool = False,
        stream_persisted: bool = False,
        allowed_tools: Optional[list[str]] = None,
        llm_kwargs: Optional[dict] = None,
    ):
        self._kernel = kernel
        self._run_context = run_context
        self._llm_client = llm_client
        self._input_data = input_data
        self._response_model = response_model
        self._agent_service = agent_service
        self._task_logger = task_logger
        self._streamer = streamer
        self._temperature = temperature
        self._stream_updates = stream_updates
        self._stream_output = stream_output
        self._stream_persisted = stream_persisted
        self._allowed_tools = allowed_tools
        self._llm_kwargs = llm_kwargs or {}
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
    ) -> tuple[ToolCall, dict, any, float, Optional[list[str]]]:
        """Resolves arguments and executes a single tool call.

        Returns: (tool_call, args, tool_result, duration, errors)
        """
        duration = 0.0
        errors = []

        def parse_json(field_name: str, value: str) -> dict:
            if not value:
                return {}
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                error_msg = f"Failed to parse {field_name} as JSON: {value}"
                logger.error(error_msg)
                errors.append(error_msg)
                return {}

        literal_args = parse_json("literal_args", tool_call.literal_args)

        planner_context_keys = parse_json(
            "planner_context_args", tool_call.planner_context_args
        )
        planner_args = {
            arg_name: self._planner_context.get(ctx_key)
            for arg_name, ctx_key in planner_context_keys.items()
        }

        input_keys = parse_json("input_args", tool_call.input_args)

        input_args = {
            arg_name: self._input_data.get(input_key)
            for arg_name, input_key in input_keys.items()
        }

        # Check for duplicate keys
        all_keys = (
            list(literal_args.keys())
            + list(planner_args.keys())
            + list(input_args.keys())
        )
        duplicates = {k for k in all_keys if all_keys.count(k) > 1}
        if duplicates:
            error_msg = f"Duplicate argument names found in ToolCall: {duplicates}"
            logger.error(error_msg)
            errors.append(error_msg)
            return tool_call, {}, f"Error: {error_msg}", 0.0, errors

        args = {**literal_args, **planner_args, **input_args}

        # Still resolve templates in literal_args if any (backward compatibility or flexible usage)
        args = self._resolve_template(args)

        logger.info(f"Calling tool {tool_call.name} with {args}")
        tool_result = None
        try:
            start_time = time.perf_counter()
            tool_result = await self._kernel.call_tool(
                tool_uri=tool_call.name,
                arguments=args,
            )
            duration = time.perf_counter() - start_time
        except Exception as e:
            duration = time.perf_counter() - start_time
            error_msg = f"Tool {tool_call.name} failed: {e}"
            logger.error(error_msg)
            errors.append(str(e))
            tool_result = f"Error: {e}"

        return tool_call, args, tool_result, duration, errors

    async def _build_system_prompt(
        self, task: str, iter_no: int, max_iterations: int
    ) -> str:
        prompt_parts = [
            "You are a planning agent. Your goal is to achieve the following task:",
            task.strip(),
            "# Tool calling instructions:",
            "Use tools to fulfill the task.",
            "To call a tool, provide its name and arguments categorized by their source:",
            "- `literal_args`: A JSON string of literal values for the tool.",
            "- `planner_context_args`: A JSON string mapping tool argument names to keys in `planner_context` (results of previous tool calls).",
            "- `input_args`: A JSON object mapping tool argument names to keys in the provided # Inputs.",
            "",
            "Example:",
            "{",
            '  "name": "python://websearch.serper_web_search",',
            '  "literal_args": "{\\"query\\": \\"Kaval AI\\"}",',
            '  "call_id": "search_result"',
            "}",
            "",
            "Example with context and inputs:",
            "{",
            '  "name": "python://data.process",',
            '  "planner_context_args": "{\\"raw_data\\": \\"search_result\\"}",',
            '  "input_args": {"user_id": "current_user_id"},',
            '  "literal_args": "{\\"mode\\": \\"fast\\"}"',
            "}",
            "",
            "IMPORTANT: An argument name must only appear in ONE of the categories. Duplicates will cause an error.",
            "",
            "You can still use template strings in `literal_args` if needed:",
            "- {{context.key}}: Reference a result from a previous tool call via call_id.",
            "- {{input.key}}: Reference data from the provided # Inputs section.",
            "",
            "# Available tools:",
            await self._kernel.get_tool_descriptions(self._allowed_tools),
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
            "Planning data",
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
        return "\n".join(prompt_parts)

    async def _get_llm_response(
        self, messages: list[dict], response_model: Type[BaseModel]
    ) -> tuple[BaseModel, dict]:
        result, stats = await self._llm_client.chat_completions(
            messages=messages,
            response_model=response_model,
            streamer=self._streamer,
            temperature=self._temperature,
            **self._llm_kwargs,
        )

        if self._agent_service:
            try:
                await self._agent_service.add_model_call_stats(
                    stats=stats, agent_id=self._run_context.agent_id
                )
            except Exception as e:
                logger.error(f"Failed to store LLM stats: {e}")

        if self._streamer and self._stream_updates:
            await self._streamer.stream_complete(
                result.short_explanation, name="running_task"
            )

        return result, stats

    async def _process_tool_calls(self, tool_calls: list[ToolCall]):
        results = await asyncio.gather(*[self._call_tool(tc) for tc in tool_calls])

        for tool_call, args, tool_result, duration, errors in results:
            if tool_call.call_id:
                self._planner_context[tool_call.call_id] = tool_result

            if tool_call.persist_to:
                self._run_context.data[tool_call.persist_to] = tool_result
                if self._streamer and self._stream_persisted:
                    stream_value = to_plain(tool_result)
                    if not isinstance(stream_value, str):
                        stream_value = json.dumps(stream_value)
                    await self._streamer.stream_complete(
                        stream_value,
                        name=tool_call.persist_to,
                    )

            if self._task_logger:
                await self._task_logger.log_tool_call(
                    tool_uri=tool_call.name,
                    arguments=args,
                    output=tool_result,
                    duration=duration,
                    errors=errors,
                )

    async def run(
        self,
        task_name: str,
        task: str,
        chat_history: list[dict] = None,
        max_iterations: int = 10,
    ):
        if chat_history is None:
            chat_history = []

        StepOutput = get_step_output_type(self._response_model)
        final_output = None
        start_time = time.perf_counter()
        initial_system_prompt = None  # Capture the first system prompt for logging

        for iter_no in range(max_iterations):
            system_prompt = await self._build_system_prompt(
                task, iter_no, max_iterations
            )

            # Capture the initial system prompt for logging
            if iter_no == 0:
                initial_system_prompt = system_prompt

            logger.info(f"Running iteration {iter_no} for task: {task_name}")

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "Analyze the situation and provide the next step output.",
                },
            ] + chat_history

            step_output, stats = await self._get_llm_response(messages, StepOutput)
            self._step_outputs.append(step_output)

            logger.info(f"Step {iter_no}: {step_output.short_explanation}")
            logger.info(f"Tool calls {iter_no}: {step_output.tool_calls}")

            if step_output.tool_calls:
                await self._process_tool_calls(step_output.tool_calls)

            if step_output.output is not None:
                final_output = step_output.output
                if not step_output.tool_calls:
                    break

        # Log the overall agent task with the initial system prompt
        duration = time.perf_counter() - start_time
        if self._task_logger and initial_system_prompt:
            await self._task_logger.log_agent_task(
                task_name=task_name,
                system_prompt=initial_system_prompt,
                input_data=self._input_data,
                output=to_plain(final_output) if final_output else None,
                duration=duration,
            )

        if final_output is not None:
            if self._streamer and self._stream_output:
                await self._streamer.stream_complete(final_output.model_dump_json())
            return final_output

        return None
