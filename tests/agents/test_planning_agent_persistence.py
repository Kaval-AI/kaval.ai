import pytest
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel, Field
from kavalai.agents.planning_agent import PlanningAgent, get_step_output_type, ToolCall
from kavalai.functionkernel import FunctionKernel
from kavalai.agents.run_context import RunContext
from kavalai.llm_clients.llm_client import LLMClient


class MockResponse(BaseModel):
    tool_result: str = Field(description="The result of the tool call")
    other_field: str = Field(default="fixed")


@pytest.mark.asyncio
async def test_planning_agent_auto_persistence():
    # Setup mocks
    kernel = MagicMock(spec=FunctionKernel)
    kernel.get_tool_descriptions = AsyncMock(return_value="Mock Tool Description")
    kernel.call_tool = AsyncMock(return_value="REAL_TOOL_OUTPUT")

    run_context = MagicMock(spec=RunContext)
    llm_client = MagicMock(spec=LLMClient)

    input_data = {}
    task = "Use tool and return result"

    agent = PlanningAgent(
        kernel=kernel,
        run_context=run_context,
        llm_client=llm_client,
        input_data=input_data,
        response_model=MockResponse,
    )

    StepOutput = get_step_output_type(MockResponse)

    # First iteration: call tool with call_id="tool_result"
    step1 = StepOutput(
        short_explanation="Calling tool",
        long_explanation="I will call the tool to get some data",
        step_number=0,
        max_steps=2,
        tool_calls=[
            ToolCall(name="python://test_tool", call_id="tool_result", args="{}")
        ],
        output=None,
    )

    # Second iteration: return output with tool_result field EMPTY (None)
    # The expectation is that PlanningAgent will fill it from planner_context["tool_result"]
    step2 = StepOutput(
        short_explanation="Returning result",
        long_explanation="I am returning the result I got from the tool",
        step_number=1,
        max_steps=2,
        tool_calls=[],
        output=MockResponse(tool_result="", other_field="from_llm"),
    )

    # Third case: type mismatch
    agent_mismatch = PlanningAgent(
        kernel=kernel,
        run_context=run_context,
        llm_client=llm_client,
        input_data=input_data,
        response_model=MockResponse,
    )
    agent_mismatch._planner_context = {"tool_result": 123}  # Int instead of string

    step_mismatch = StepOutput(
        short_explanation="Returning result",
        long_explanation="I am returning the result",
        step_number=0,
        max_steps=1,
        tool_calls=[],
        output=MockResponse(tool_result="", other_field="mismatch"),
    )

    llm_client.chat_completions.side_effect = [
        (step1, {"stats": "dummy"}),
        (step2, {"stats": "dummy"}),
        (step_mismatch, {"stats": "dummy"}),
    ]

    # Run agent
    result = await agent.run(task=task, max_iterations=2)

    # Assertions
    assert result is not None
    assert result.other_field == "from_llm"
    # This is what we want to achieve:
    assert (
        result.tool_result == "REAL_TOOL_OUTPUT"
    ), "Value should be auto-persisted from planner_context"

    # Run agent for mismatch
    result_mismatch = await agent_mismatch.run(task=task, max_iterations=1)
    assert result_mismatch is not None
    assert (
        result_mismatch.tool_result == ""
    )  # Should NOT be updated due to type mismatch
    assert result_mismatch.other_field == "mismatch"
