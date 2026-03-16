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


@pytest.mark.asyncio
async def test_planning_agent_auto_persistence_after_all_steps():
    """
    Test that auto-persistence is done AFTER all steps, which means even if a tool
    is called in the same step as the output, its result should be available.
    Current implementation might fail if it checks for output BEFORE updating planner_context
    or if it returns too early.
    """
    kernel = MagicMock(spec=FunctionKernel)
    kernel.get_tool_descriptions = AsyncMock(return_value="Mock Tool Description")
    kernel.call_tool = AsyncMock(return_value="LATE_TOOL_OUTPUT")

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

    # In one single step, LLM provides BOTH a tool call and an output.
    # Auto-persistence should ensure the output's "tool_result" field is filled with "LATE_TOOL_OUTPUT".
    step1 = StepOutput(
        short_explanation="Doing both",
        long_explanation="I am calling a tool and returning output in the same step",
        step_number=0,
        max_steps=1,
        tool_calls=[
            ToolCall(name="python://test_tool", call_id="tool_result", args="{}")
        ],
        output=MockResponse(tool_result="", other_field="simultaneous"),
    )

    llm_client.chat_completions.side_effect = [
        (step1, {"stats": "dummy"}),
    ]

    result = await agent.run(task="do both", max_iterations=1)

    assert result is not None
    # IF this is "REAL_TOOL_OUTPUT" it means it used a STALE result (though in this case none exists)
    # Actually, current code SHOULD fail this test if it processes output BEFORE tools in the same step.
    # Wait, current code processes tools first? Let's check.
    assert result.tool_result == "LATE_TOOL_OUTPUT"


@pytest.mark.asyncio
async def test_planning_agent_multiple_outputs():
    """
    Test that auto-persistence only happens at the end, and we use the LAST output provided.
    Currently, the agent returns as soon as it sees ANY output.
    """
    kernel = MagicMock(spec=FunctionKernel)
    kernel.get_tool_descriptions = AsyncMock(return_value="Mock Tool Description")
    kernel.call_tool = AsyncMock(return_value="SECOND_RESULT")

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

    # Step 1: provides an output but also a tool call for the NEXT output
    step1 = StepOutput(
        short_explanation="Providing premature output",
        long_explanation="I have an output but I'm also calling a tool to improve it later",
        step_number=0,
        max_steps=2,
        tool_calls=[
            ToolCall(name="python://test_tool", call_id="tool_result", args="{}")
        ],
        output=MockResponse(tool_result="premature", other_field="step1"),
    )

    # Step 2: provides final output
    step2 = StepOutput(
        short_explanation="Providing final output",
        long_explanation="Now I have the full result",
        step_number=1,
        max_steps=2,
        tool_calls=[],
        output=MockResponse(tool_result="", other_field="step2"),
    )

    llm_client.chat_completions.side_effect = [
        (step1, {"stats": "dummy"}),
        (step2, {"stats": "dummy"}),
    ]

    result = await agent.run(task="multiple outputs", max_iterations=2)

    # Current implementation will return after step1 with "premature" (or updated "SECOND_RESULT" if tools processed first)
    # But if we want it to complete all steps or at least handle it better:
    # Actually, if it returns, it stops.
    assert result.other_field == "step2"
    assert result.tool_result == "SECOND_RESULT"


@pytest.mark.asyncio
async def test_planning_agent_auto_persistence_off():
    # Setup mocks
    kernel = MagicMock(spec=FunctionKernel)
    kernel.get_tool_descriptions = AsyncMock(return_value="Mock Tool Description")
    kernel.call_tool = AsyncMock(return_value="REAL_TOOL_OUTPUT")

    run_context = MagicMock(spec=RunContext)
    llm_client = MagicMock(spec=LLMClient)

    input_data = {}
    task = "Use tool and return result"

    # auto_persist=False
    agent = PlanningAgent(
        kernel=kernel,
        run_context=run_context,
        llm_client=llm_client,
        input_data=input_data,
        response_model=MockResponse,
        auto_persist=False,
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

    # Second iteration: return output with tool_result field EMPTY
    # Since auto_persist is False, it should NOT be filled.
    step2 = StepOutput(
        short_explanation="Returning result",
        long_explanation="I am returning the result",
        step_number=1,
        max_steps=2,
        tool_calls=[],
        output=MockResponse(tool_result="original", other_field="from_llm"),
    )

    llm_client.chat_completions.side_effect = [
        (step1, {"stats": "dummy"}),
        (step2, {"stats": "dummy"}),
    ]

    # Run agent
    result = await agent.run(task=task, max_iterations=2)

    # Assertions
    assert result is not None
    assert (
        result.tool_result == "original"
    ), "Value should NOT be auto-persisted when auto_persist=False"

    # Check that system prompt DOES NOT contain auto-persistence instructions
    llm_client.chat_completions.assert_called()
    for call in llm_client.chat_completions.call_args_list:
        messages = call.kwargs["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")["content"]
        assert "automatically copied to the matching field" not in system_msg
