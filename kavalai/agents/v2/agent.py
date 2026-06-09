import asyncio
import json
from typing import Any, Optional, Type
import os

from loguru import logger
from pydantic import BaseModel, Field, ConfigDict

from kavalai.agents.run_context import RunContext
from kavalai.agents.utils import to_plain
from kavalai.functionkernel import FunctionKernel
from kavalai.llm_clients.base_client import BaseLlmClient, ChatHistory, ChatMessage
from jinja2 import Template


class ToolCall(BaseModel):
    """This data structure represents tool call requests.

    Arguments are expected to be JSON encoded to help LLM models encode the data.
    """

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


def get_step_output_type(ResponseModel=Type[BaseModel]):
    class StepOutput(BaseModel):
        """Data structure that helps passing around information between consecutive agent runs."""

        instructions: str = Field(
            description="Briefly describe the goal of this step (what you intend to achieve with these tool calls).",
        )
        tool_calls: list[ToolCall] = Field(
            default=[],
            description="Add tool call requests here, their output will be available via `call_id` key for next steps.",
        )
        output: Optional[ResponseModel] = None

    return StepOutput


class Agent:
    def __init__(
        self,
        llm_client: BaseLlmClient,
        *,
        kernel: Optional[FunctionKernel] = None,
        run_context: Optional[RunContext] = None,
        prompt_template: Optional[Template] = None,
        debug: bool = False,
    ):
        self.debug = debug
        self.kernel = kernel
        if not run_context:
            run_context = RunContext()
        self.run_context = run_context
        self.llm_client = llm_client
        if prompt_template is None:
            with open(
                os.path.join(os.path.dirname(__file__), "default_prompt_template.j2"),
                "r",
            ) as f:
                prompt_template = Template(f.read())
        self.prompt_template = prompt_template

    async def prompt(
        self,
        prompt: str,
        response_model: Optional[Type[BaseModel]] = None,
        max_steps: int = 10,
    ) -> str | BaseModel:
        """Run the agent loop, calling tools until it produces a final output.

        The agent iterates up to ``max_steps`` times. On each step the LLM
        returns a ``StepOutput`` with optional ``tool_calls`` and an optional
        final ``output``. Tool calls are executed through the
        ``FunctionKernel`` and their results are fed back into the prompt so
        the model can reason over them on the next step. The loop stops once
        the model returns an ``output`` without requesting further tool calls,
        or when ``max_steps`` is reached.

        Args:
            prompt: The task description for the agent.
            response_model: Optional Pydantic model describing the structured
                final output. When omitted, a plain string is returned.
            max_steps: Maximum number of reasoning/tool-calling iterations.

        Returns:
            The structured ``response_model`` instance, or a string when no
            ``response_model`` is provided. ``None`` if no output was produced.
        """
        StepOutput = get_step_output_type(response_model or str)

        # Per-invocation working memory: tool call results keyed by call_id,
        # referenced via `planner_context_args`. Created fresh for each
        # `prompt()` call (up to `max_steps`) and discarded afterwards, unlike
        # `self.run_context` which is passed in at construction.
        planner_context: dict[str, Any] = {}
        # Record of executed steps, rendered back into the prompt template.
        steps: list[dict] = []
        final_output: Optional[BaseModel] = None

        for step_idx in range(max_steps):
            rendered_prompt = self.prompt_template.render(
                prompt=prompt,
                data=self.run_context.data,
                tool_descriptions=(
                    await self.kernel.get_tool_descriptions() if self.kernel else ""
                ),
                steps=steps,
                current_step=step_idx,
                max_steps=max_steps,
            )

            if self.debug:
                print(rendered_prompt)

            chat_history = ChatHistory(
                messages=[
                    ChatMessage(role="system", content=rendered_prompt),
                    ChatMessage(
                        role="user",
                        content="Analyze the situation and provide the next step output.",
                    ),
                ]
            )

            logger.info(f"Agent step {step_idx}/{max_steps}")
            step_output = await self.llm_client.chat_completions(
                chat_history=chat_history, response_model=StepOutput
            )

            if step_output is None:
                logger.warning("LLM returned no step output, stopping.")
                break

            # Ensure every tool call has a stable id for context lookups.
            for idx, tool_call in enumerate(step_output.tool_calls):
                if not tool_call.call_id:
                    tool_call.call_id = f"tool_call_{step_idx}_{idx}"

            step_record: dict[str, Any] = {
                "index": step_idx,
                "instructions": step_output.instructions,
                "tool_calls": [],
                "output": to_plain(step_output.output)
                if step_output.output is not None
                else None,
            }

            if step_output.tool_calls and self.kernel:
                results = await asyncio.gather(
                    *[
                        self._call_tool(tc, planner_context)
                        for tc in step_output.tool_calls
                    ]
                )
                for tool_call, args, result in results:
                    planner_context[tool_call.call_id] = result
                    step_record["tool_calls"].append(
                        {
                            "name": tool_call.name,
                            "args": args,
                            "call_id": tool_call.call_id,
                            "output": to_plain(result),
                        }
                    )

            steps.append(step_record)

            if step_output.output is not None:
                final_output = step_output.output
                # Stop once the model produced an answer without more tool calls.
                if not step_output.tool_calls:
                    break

        return final_output

    def _resolve_args(
        self, tool_call: ToolCall, planner_context: dict[str, Any]
    ) -> dict:
        """Resolve a ToolCall's argument sources into a single argument dict.

        Arguments are merged with precedence ``literal_args`` >
        ``planner_context_args`` > ``input_args``. ``planner_context_args``
        resolves against the per-invocation ``planner_context`` (results of
        previous tool calls); ``input_args`` against ``self.run_context.data``.
        """

        def parse(field_name: str, value: str) -> dict:
            if not value:
                return {}
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse {field_name} as JSON: {value}")
                return {}

        literal_args = parse("literal_args", tool_call.literal_args)

        context_keys = parse("planner_context_args", tool_call.planner_context_args)
        context_args = {
            arg_name: planner_context.get(ctx_key)
            for arg_name, ctx_key in context_keys.items()
        }

        input_keys = parse("input_args", tool_call.input_args)
        input_args = {
            arg_name: self.run_context.data.get(input_key)
            for arg_name, input_key in input_keys.items()
        }

        return {**input_args, **context_args, **literal_args}

    async def _call_tool(
        self, tool_call: ToolCall, planner_context: dict[str, Any]
    ) -> tuple[ToolCall, dict, Any]:
        """Resolve arguments and execute a single tool call via the kernel.

        Returns ``(tool_call, args, result)``. Execution errors are captured
        and returned as the result so the model can self-correct.
        """
        args = self._resolve_args(tool_call, planner_context)
        logger.info(f"Calling tool {tool_call.name} with {args}")
        try:
            result = await self.kernel.call_tool(
                tool_uri=tool_call.name, arguments=args
            )
        except Exception as e:
            logger.error(f"Tool {tool_call.name} failed: {e}")
            result = f"Error: {e}"
        return tool_call, args, result


if __name__ == "__main__":
    from kavalai.llm_clients.v2.openai_client import OpenAIClient
    from kavalai.functionkernel import FunctionKernel, pythontool
    import datetime

    @pythontool
    def get_time():
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    kernel = FunctionKernel()
    kernel.register_python_tool("get_time", get_time)

    llm_client = OpenAIClient("gpt-5.4-mini")
    agent = Agent(llm_client=llm_client, kernel=kernel, debug=True)
    result = asyncio.run(agent.prompt("Greet the user based on current time!"))
    print(result)
