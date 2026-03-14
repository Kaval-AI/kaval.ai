import pytest
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel
from kavalai.agents.planning_agent import PlanningAgent, ToolCall, get_step_output_type
from kavalai.functionkernel import FunctionKernel
from kavalai.agents.run_context import RunContext
from kavalai.llm_clients.llm_client import LLMClient


class MockResponse(BaseModel):
    answer: str


@pytest.mark.asyncio
async def test_planning_agent_run_success():
    # Setup mocks
    kernel = MagicMock(spec=FunctionKernel)
    kernel.get_tool_descriptions = AsyncMock(return_value="Mock Tool Description")
    kernel.call_tool = AsyncMock(return_value="Tool Result")

    run_context = MagicMock(spec=RunContext)
    llm_client = MagicMock(spec=LLMClient)

    input_data = {"key": "value"}
    task = "Solve this task"

    agent = PlanningAgent(
        kernel=kernel,
        run_context=run_context,
        llm_client=llm_client,
        input_data=input_data,
        response_model=MockResponse,
    )

    StepOutput = get_step_output_type(MockResponse)

    # Mock LLM response for first iteration (tool call)
    step1 = StepOutput(
        short_explanation="Step 1",
        long_explanation="Planning to call a tool",
        step_number=0,
        max_steps=2,
        tool_calls=[ToolCall(name="python://tool", call_id="call1", args="{}")],
        output=None,
    )

    # Mock LLM response for second iteration (final answer)
    step2 = StepOutput(
        short_explanation="Step 2",
        long_explanation="Finished",
        step_number=1,
        max_steps=2,
        tool_calls=[],
        output=MockResponse(answer="Final Answer"),
    )

    llm_client.chat_completions.side_effect = [
        (step1, {"stats": "dummy"}),
        (step2, {"stats": "dummy"}),
    ]

    # Run agent
    result = await agent.run(task=task, max_iterations=5)

    # Assertions
    assert isinstance(result, MockResponse)
    assert result.answer == "Final Answer"
    assert len(agent._step_outputs) == 2
    assert agent._planner_context["call1"] == "Tool Result"

    assert llm_client.chat_completions.call_count == 2
    kernel.call_tool.assert_awaited_once_with(tool_uri="python://tool", arguments={})


@pytest.mark.asyncio
async def test_planning_agent_max_iterations():
    kernel = MagicMock(spec=FunctionKernel)
    kernel.get_tool_descriptions = AsyncMock(return_value="")
    run_context = MagicMock(spec=RunContext)
    llm_client = MagicMock(spec=LLMClient)

    agent = PlanningAgent(
        kernel=kernel,
        run_context=run_context,
        llm_client=llm_client,
        input_data={},
        response_model=MockResponse,
    )

    StepOutput = get_step_output_type(MockResponse)
    step = StepOutput(
        short_explanation="Wait",
        long_explanation="Waiting...",
        step_number=0,
        max_steps=10,
        tool_calls=[],
        output=None,
    )

    llm_client.chat_completions.return_value = (step, {})

    result = await agent.run(task="task", max_iterations=3)

    assert result is None
    assert llm_client.chat_completions.call_count == 3
