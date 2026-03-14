import pytest
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel
from kavalai.agents.planning_agent import PlanningAgent, get_step_output_type, ToolCall
from kavalai.functionkernel import FunctionKernel


class MockResponse(BaseModel):
    answer: str


@pytest.mark.asyncio
async def test_planning_agent_tool_failure_handling():
    # Setup mocks
    kernel = MagicMock(spec=FunctionKernel)
    kernel.get_tool_descriptions = AsyncMock(return_value="Mock Tool Description")

    # Mock tool call failure
    kernel.call_tool = AsyncMock(
        side_effect=Exception("Validation error: field required")
    )

    llm_client = MagicMock()
    llm_client.chat_completions = AsyncMock()

    agent = PlanningAgent(
        kernel=kernel,
        run_context=MagicMock(),
        llm_client=llm_client,
        input_data={},
        response_model=MockResponse,
    )

    StepOutput = get_step_output_type(MockResponse)

    # First step: LLM calls a tool (which will fail)
    step1 = StepOutput(
        short_explanation="Calling tool",
        long_explanation="Need to call a tool",
        step_number=0,
        max_steps=2,
        tool_calls=[ToolCall(name="python://fail", call_id="c1", args={})],
        output=None,
    )

    # Second step: LLM provides final output based on (failed) tool result
    final_output = MockResponse(answer="Handled failure")
    step2 = StepOutput(
        short_explanation="Done",
        long_explanation="Finished the task",
        step_number=1,
        max_steps=2,
        tool_calls=[],
        output=final_output,
    )

    llm_client.chat_completions.side_effect = [(step1, {}), (step2, {})]

    # Run agent
    result = await agent.run(task="task", max_iterations=2)

    # Assertions
    assert result == final_output
    assert kernel.call_tool.called

    # Check if the error message was put into planner_context
    assert "c1" in agent._planner_context
    assert "Error: Validation error: field required" in agent._planner_context["c1"]
