"""
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import json
import time
from typing import Type, Optional, Any, Dict, List, Tuple

from loguru import logger
from pydantic import BaseModel, Field, ConfigDict

from kavalai.agents.agent_service import AgentService
from kavalai.agents.run_context import RunContext
from kavalai.agents.task_logger import TaskLogger
from kavalai.functionkernel import FunctionKernel
from kavalai.llm_clients.base_client import (
    BaseLlmClient,
    ChatHistory,
    ChatMessage,
    ModelCallStat,
    ModelStatsReceiver,
)
from kavalai.llm_clients.common import Streamer
from kavalai.agents.utils import to_plain


class ToolCall(BaseModel):
    """This data structure represents tool call requests."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Tool call name i.e python://mypackage.mytool")
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


def get_step_output_type(ResponseModel: Type[BaseModel] = BaseModel):
    class StepOutput(BaseModel):
        """Data structure that helps passing around information between consecutive agent runs."""

        short_explanation: str = Field(
            description="Human friendly summary of planned steps. Be very concise, maximum 50 characters.",
        )
        instructions: str = Field(
            description="Store very brief instructions for downstream LLM.",
        )
        tool_calls: List[ToolCall] = Field(
            default=[],
            description="Add tool call requests here, their output will be stored in planner_context accessible with `call_id` key.",
        )
        output: Optional[ResponseModel] = None

    return StepOutput


class AgentTemplater:
    """Handles resolution of template strings in tool arguments."""

    def __init__(self, planner_context: Dict[str, Any], input_data: Dict[str, Any]):
        self.planner_context = planner_context
        self.input_data = input_data

    def resolve(self, value: Any) -> Any:
        """Resolves template strings like {{context.key}} or {{input.key}}."""
        if isinstance(value, str):
            if value.startswith("{{context.") and value.endswith("}}"):
                key = value[10:-2].strip()
                return self.planner_context.get(key, value)
            elif value.startswith("{{input.") and value.endswith("}}"):
                key = value[8:-2].strip()
                return self.input_data.get(key, value)
        elif isinstance(value, dict):
            return {k: self.resolve(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.resolve(v) for v in value]
        return value


class Agent(ModelStatsReceiver):
    def __init__(
        self,
        *,
        kernel: FunctionKernel,
        run_context: RunContext,
        llm_client: BaseLlmClient,
        input_data: Dict[str, Any],
        response_model: Type[BaseModel] = BaseModel,
        agent_service: Optional[AgentService] = None,
        task_logger: Optional[TaskLogger] = None,
        streamer: Optional[Streamer] = None,
        stream_updates: bool = False,
        stream_output: bool = False,
        stream_persisted: bool = False,
        allowed_tools: Optional[List[str]] = None,
    ):
        self._kernel = kernel
        self._run_context = run_context
        self._llm_client = llm_client
        self._input_data = input_data
        self._response_model = response_model
        self._agent_service = agent_service
        self._task_logger = task_logger
        self._streamer = streamer
        self._stream_updates = stream_updates
        self._stream_output = stream_output
        self._stream_persisted = stream_persisted
        self._allowed_tools = allowed_tools

        self._planner_context = {}
        self._step_outputs = []
        self._templater = AgentTemplater(self._planner_context, self._input_data)

        # Set this agent as the receiver for LLM stats
        if hasattr(self._llm_client, "model_stats_receiver"):
            self._llm_client.model_stats_receiver = self

    def receive_model_stats(self, stats: ModelCallStat):
        """Receive and store LLM call statistics."""
        if self._agent_service:
            # Note: Using create_task because this is usually called from an async context
            # but we don't want to block the LLM client.
            asyncio.create_task(
                self._agent_service.add_model_call_stats(
                    stats=stats, agent_id=self._run_context.agent_id
                )
            )

    async def _build_system_prompt(
        self, task: str, iter_no: int, max_iterations: int
    ) -> str:
        prompt_parts = [
            "You are a planning agent. Your goal is to achieve the following task:",
            task.strip(),
            "[TOOL CALLING INSTRUCTIONS]",
            "Use tools to fulfill the task.",
            "To call a tool, provide its name and arguments categorized by their source:",
            "- `literal_args`: A JSON string of literal values for the tool (e.g., constant values, empty arrays).",
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
            '  "input_args": "{\\"user_id\\": \\"current_user_id\\"}",',
            '  "literal_args": "{\\"mode\\": \\"fast\\"}"',
            "}",
            "",
            "CRITICAL RULE: Each parameter name MUST appear in ONLY ONE of literal_args, planner_context_args, or input_args.",
            '- Use literal_args for constant values (e.g., {"countries": []})',
            "- Use planner_context_args for references to previous tool results",
            "- Use input_args for references to input data",
            "- NEVER specify the same parameter in multiple categories",
            "",
            "Argument Precedence (when merging): literal_args > planner_context_args > input_args",
            "While duplicates are merged with this precedence, they indicate an error in your planning and will be logged.",
            "",
            "You can still use template strings in `literal_args` if needed:",
            "- {{context.key}}: Reference a result from a previous tool call via call_id.",
            "- {{input.key}}: Reference data from the provided # Inputs section.",
            "",
            "[AVAILABLE TOOLS]",
            await self._kernel.get_tool_descriptions(self._allowed_tools),
            "[INPUTS]",
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
            "[PLANNING DATA]",
            f"current_step={iter_no}",
            f"max_steps={max_iterations}",
            "",
            "[STEP OUTPUTS (PREVIOUS STEPS)]",
            json.dumps(
                [
                    so.model_dump() if hasattr(so, "model_dump") else so
                    for so in self._step_outputs
                ],
                indent=2,
            ),
        ]
        return "\n".join(prompt_parts)

    def _parse_json_arg(
        self, field_name: str, value: str, errors: List[str]
    ) -> Dict[str, Any]:
        """Safely parse a JSON string argument."""
        if not value:
            return {}
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            error_msg = f"Failed to parse {field_name} as JSON: {value}"
            logger.error(error_msg)
            errors.append(error_msg)
            return {}

    def _resolve_tool_arguments(
        self, tool_call: ToolCall
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Resolve all argument sources for a tool call."""
        errors = []

        literal_args = self._parse_json_arg(
            "literal_args", tool_call.literal_args, errors
        )

        context_keys = self._parse_json_arg(
            "planner_context_args", tool_call.planner_context_args, errors
        )
        planner_args = {
            arg_name: self._planner_context.get(ctx_key)
            for arg_name, ctx_key in context_keys.items()
        }

        input_keys = self._parse_json_arg("input_args", tool_call.input_args, errors)
        input_args = {}
        for arg_name, input_key in input_keys.items():
            if not isinstance(input_key, str):
                error_msg = (
                    f"Invalid input_args for '{arg_name}': expected a string key referencing input_data, "
                    f"but got {type(input_key).__name__}: {input_key}."
                )
                logger.error(error_msg)
                errors.append(error_msg)
                continue
            input_args[arg_name] = self._input_data.get(input_key)

        # Check for duplicates
        all_arg_names = (
            list(literal_args.keys())
            + list(planner_args.keys())
            + list(input_args.keys())
        )
        duplicates = {k for k in all_arg_names if all_arg_names.count(k) > 1}
        if duplicates:
            error_msg = f"Duplicate argument names found in ToolCall: {duplicates}. Precedence: literal > context > input"
            logger.error(error_msg)
            errors.append(error_msg)

        # Merge with precedence
        args = {**input_args, **planner_args, **literal_args}

        # Resolve any nested templates
        args = self._templater.resolve(args)

        return args, errors

    async def _execute_tool(
        self, tool_uri: str, arguments: Dict[str, Any]
    ) -> Tuple[Any, float, Optional[str]]:
        """Execute the tool via the kernel."""
        start_time = time.perf_counter()
        try:
            result = await self._kernel.call_tool(
                tool_uri=tool_uri, arguments=arguments
            )
            duration = time.perf_counter() - start_time
            return result, duration, None
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(f"Tool {tool_uri} failed: {e}")
            return f"Error: {e}", duration, str(e)

    async def _call_tool(
        self, tool_call: ToolCall
    ) -> Tuple[ToolCall, Dict[str, Any], Any, float, List[str]]:
        """Higher-level tool call handler that resolves args and executes."""
        args, errors = self._resolve_tool_arguments(tool_call)

        logger.info(f"Calling tool {tool_call.name} with {args}")
        result, duration, exec_error = await self._execute_tool(tool_call.name, args)

        if exec_error:
            errors.append(exec_error)

        return tool_call, args, result, duration, errors

    async def _get_llm_response(
        self, chat_history: ChatHistory, response_model: Type[BaseModel]
    ) -> Any:
        """Get structured response from the v2 LLM client, with optional token streaming."""
        if self._streamer:
            # Bridge v2 LLM streamer to our common.Streamer for token-by-token updates
            llm_streamer = await self._llm_client.stream_chat_completions(
                chat_history=chat_history,
                response_model=response_model,
            )
            final_json = None
            async for chunk in llm_streamer:
                if chunk.type == "partial":
                    name = None
                    if chunk.name == "thought":
                        name = f"{self._streamer.name}_thought"
                    await self._streamer.stream_partial(chunk.value, name=name)
                elif chunk.type == "thought_partial":
                    await self._streamer.stream_partial(
                        chunk.value, name=f"{self._streamer.name}_thought"
                    )
                elif chunk.type == "complete":
                    final_json = chunk.value

            if final_json:
                result = response_model.model_validate_json(final_json)
            else:
                result = None
        else:
            result = await self._llm_client.chat_completions(
                chat_history=chat_history,
                response_model=response_model,
            )

        if (
            self._streamer
            and self._stream_updates
            and hasattr(result, "short_explanation")
        ):
            await self._streamer.stream_complete(
                result.short_explanation, name="running_task"
            )

        return result

    async def _process_tool_calls(self, tool_calls: List[ToolCall]):
        """Process multiple tool calls in parallel."""
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
                        stream_value, name=tool_call.persist_to
                    )

            if self._task_logger:
                self._task_logger.log_tool_call(
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
        chat_history: Optional[List[Dict[str, Any]]] = None,
        max_iterations: int = 10,
    ) -> Optional[Any]:
        """Main execution loop for the agent."""
        if chat_history is None:
            chat_history = []

        # Convert simple dict history to ChatHistory/ChatMessage if needed
        # (Though we mostly just append to it in this loop)

        StepOutput = get_step_output_type(self._response_model)
        final_output = None
        start_time = time.perf_counter()
        initial_system_prompt = None

        for iter_no in range(max_iterations):
            step_start_time = time.perf_counter()
            system_prompt = await self._build_system_prompt(
                task, iter_no, max_iterations
            )

            if iter_no == 0:
                initial_system_prompt = system_prompt

            logger.info(f"Running iteration {iter_no} for task: {task_name}")

            messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(
                    role="user",
                    content="Analyze the situation and provide the next step output.",
                ),
            ]

            # Append history if any (converting dicts to ChatMessage)
            for msg in chat_history:
                messages.append(ChatMessage(**msg))

            history = ChatHistory(messages=messages)
            step_output = await self._get_llm_response(history, StepOutput)

            if step_output is None:
                logger.error("LLM returned None response")
                break

            self._step_outputs.append(step_output)

            # Assign call IDs if missing
            for idx, tool_call in enumerate(step_output.tool_calls):
                if tool_call.call_id is None:
                    tool_call.call_id = f"tool_call_{iter_no}_{idx}"

            logger.info(f"Step {iter_no}: {step_output.short_explanation}")

            if step_output.tool_calls:
                await self._process_tool_calls(step_output.tool_calls)

            if self._task_logger:
                self._task_logger.log_agent_task(
                    task_name=f"{task_name}_step_{iter_no}",
                    system_prompt=system_prompt,
                    input_data=self._input_data,
                    output=to_plain(step_output),
                    duration=time.perf_counter() - step_start_time,
                )

            if step_output.output is not None:
                final_output = step_output.output
                if not step_output.tool_calls:
                    break

        duration = time.perf_counter() - start_time
        if self._task_logger and initial_system_prompt:
            self._task_logger.log_agent_task(
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
