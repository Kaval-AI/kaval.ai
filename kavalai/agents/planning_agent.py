import logging
from typing import Type, Optional

from pydantic import BaseModel, Field

from kavalai.agents.run_context import RunContext
from kavalai.functionkernel import FunctionKernel
from kavalai.llm_clients.llm_client import LLMClient

logger = logging.getLogger(__name__)


class ToolCall(BaseModel):
    """This data structure represents tool call requests."""

    name: str = Field(description="Tool call name i.e python://mypackage.myfunc")
    call_id: Optional[str] = Field(
        description="Generate an ID, which represents this result in downstream agent runs."
    )
    args: dict = Field(description="Function arguments")


def get_step_output_type(ResponseModel=Type[BaseModel]):
    class StepOutput(BaseModel):
        """Data structure that helps passing around information between consecutive agent runs."""

        short_explanation: str = Field(
            description="Human friendly summary of planned steps.", max_length=50
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
            description="Leave output None if tool calls are given, this means output should be constructed in down-stream process."
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
    ):
        self._kernel = kernel
        self._run_context = run_context
        self._llm_client = llm_client
        self._input_data = input_data
        self._response_model = response_model
        self._planner_context = {}
        self._step_outputs = []

    async def run(
        self, task: str, chat_history: list[dict] = None, max_iterations: int = 10
    ):
        if chat_history is None:
            chat_history = []

        StepOutput = get_step_output_type(self._response_model)

        for iter_no in range(max_iterations):
            system_prompt = (
                task + "\n\n"
                f"Available tools:\n{await self._kernel.get_tool_descriptions()}\n\n"
                f"Inputs:\n{self._input_data}\n\n"
                f"Planner Context (tool results):\n{self._planner_context}\n\n"
                f"Step Outputs (previous steps):\n{[so.model_dump() for so in self._step_outputs]}\n"
            )

            messages = [{"role": "system", "content": system_prompt}] + chat_history

            result, stats = await self._llm_client.chat_completions(
                messages=messages,
                response_model=StepOutput,
            )

            step_output: StepOutput = result
            self._step_outputs.append(step_output)

            if step_output.tool_calls:
                for tool_call in step_output.tool_calls:
                    tool_result = await self._kernel.call_tool(
                        tool_uri=tool_call.name,
                        arguments=tool_call.args,
                    )
                    if tool_call.call_id:
                        self._planner_context[tool_call.call_id] = tool_result

            if step_output.output is not None:
                return step_output.output

        return None
