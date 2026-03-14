from kavalai.functionkernel import FunctionKernel, pythontool
from kavalai.agents.run_context import RunContext
from kavalai.llm_clients.llm_client import LLMClient
from typing import Type, Optional
from pydantic import BaseModel, Field
import logging

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
        response_model=Type[BaseModel],
    ):
        self._kernel = kernel
        self._run_context = run_context
        self._llm_client = llm_client
        self._should_stop = False

    async def run(
        self, task: str, message_history: list[dict] = None, max_iterations: int = 10
    ):
        for iter_no in range(max_iterations):
            if self._should_stop:
                break

    @pythontool
    async def stop(self):
        """Stops the planning agent. Agent must call this when the task is fulfilled."""
        logger.info("Stopping planning agent...")
        self._should_stop = True
