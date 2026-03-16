import pytest
import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel, Field
from kavalai.agents.planning_agent import PlanningAgent, ToolCall, get_step_output_type
from kavalai.functionkernel import FunctionKernel, pythontool
from kavalai.agents.run_context import RunContext
from kavalai.llm_clients.llm_client import LLMClient
from kavalai.llm_clients.common import Streamer


class MockResponse(BaseModel):
    answer: str


class MockPersistenceResponse(BaseModel):
    tool_result: str = Field(description="The result of the tool call")
    other_field: str = Field(default="fixed")


class MockErrorResponse(BaseModel):
    result: str


@pytest.mark.asyncio
async def test_planning_agent_run_success():
    # Setup mocks
    kernel = MagicMock(spec=FunctionKernel)
    kernel.get_tool_descriptions = AsyncMock(return_value="Mock Tool Description")
    kernel.call_tool = AsyncMock(return_value="Tool Result")

    run_context = MagicMock(spec=RunContext)
    run_context.agent_id = "agent-id"
    run_context.session_id = "session-id"
    run_context.run_id = "run-id"
    run_context.agent_service = AsyncMock()

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
        tool_calls=[ToolCall(name="python://tool", call_id="call1", args='{"a": 1}')],
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
    kernel.call_tool.assert_awaited_once_with(
        tool_uri="python://tool", arguments={"a": 1}
    )

    # Verify agent_service.add_task was called for the tool call
    run_context.agent_service.add_task.assert_awaited_once_with(
        agent_id=run_context.agent_id,
        session_id=run_context.session_id,
        run_id=run_context.run_id,
        name="python://tool",
        inputs={"arguments": {"a": 1}},
        output="Tool Result",
    )


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


@pytest.mark.asyncio
async def test_planning_agent_streaming():
    # Setup mocks
    kernel = MagicMock(spec=FunctionKernel)
    kernel.get_tool_descriptions = AsyncMock(return_value="Mock Tool Description")

    run_context = MagicMock(spec=RunContext)
    llm_client = MagicMock(spec=LLMClient)

    # Mock streamer
    queue = asyncio.Queue()
    streamer = Streamer(name="test_streamer", queue=queue)
    streamer.stream_complete = AsyncMock(wraps=streamer.stream_complete)

    input_data = {"key": "value"}
    task = "Solve this task"

    # 1. Test initialization with streamer
    agent = PlanningAgent(
        kernel=kernel,
        run_context=run_context,
        llm_client=llm_client,
        input_data=input_data,
        response_model=MockResponse,
        streamer=streamer,
        stream_updates=True,
        stream_output=True,
    )

    StepOutput = get_step_output_type(MockResponse)

    # Mock LLM response
    final_output = MockResponse(answer="Final Answer")
    step = StepOutput(
        short_explanation="Done",
        long_explanation="Finished the task",
        step_number=0,
        max_steps=1,
        tool_calls=[],
        output=final_output,
    )

    llm_client.chat_completions.return_value = (step, {"stats": "dummy"})

    # Run agent
    result = await agent.run(task=task, max_iterations=1)

    # Assertions
    assert result == final_output

    # 2. Check if streamer was passed to chat_completions
    llm_client.chat_completions.assert_called_once()
    args, kwargs = llm_client.chat_completions.call_args
    assert kwargs.get("streamer") == streamer

    # 3. Check if stream_complete was called with the final output
    streamer.stream_complete.assert_awaited()
    # It might be called multiple times (once for running_task, once for final output)
    # So we check if the last call was with final_output
    last_call = streamer.stream_complete.call_args_list[-1]
    assert last_call.args[0] == final_output.model_dump_json()


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
        response_model=MockPersistenceResponse,
    )

    StepOutput = get_step_output_type(MockPersistenceResponse)

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
        output=MockPersistenceResponse(tool_result="", other_field="from_llm"),
    )

    # Third case: type mismatch
    agent_mismatch = PlanningAgent(
        kernel=kernel,
        run_context=run_context,
        llm_client=llm_client,
        input_data=input_data,
        response_model=MockPersistenceResponse,
    )
    agent_mismatch._planner_context = {"tool_result": 123}  # Int instead of string

    step_mismatch = StepOutput(
        short_explanation="Returning result",
        long_explanation="I am returning the result",
        step_number=0,
        max_steps=1,
        tool_calls=[],
        output=MockPersistenceResponse(tool_result="", other_field="mismatch"),
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
        response_model=MockPersistenceResponse,
    )

    StepOutput = get_step_output_type(MockPersistenceResponse)

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
        output=MockPersistenceResponse(tool_result="", other_field="simultaneous"),
    )

    llm_client.chat_completions.side_effect = [
        (step1, {"stats": "dummy"}),
    ]

    result = await agent.run(task="do both", max_iterations=1)

    assert result is not None
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
        response_model=MockPersistenceResponse,
    )

    StepOutput = get_step_output_type(MockPersistenceResponse)

    # Step 1: provides an output but also a tool call for the NEXT output
    step1 = StepOutput(
        short_explanation="Providing premature output",
        long_explanation="I have an output but I'm also calling a tool to improve it later",
        step_number=0,
        max_steps=2,
        tool_calls=[
            ToolCall(name="python://test_tool", call_id="tool_result", args="{}")
        ],
        output=MockPersistenceResponse(tool_result="premature", other_field="step1"),
    )

    # Step 2: provides final output
    step2 = StepOutput(
        short_explanation="Providing final output",
        long_explanation="Now I have the full result",
        step_number=1,
        max_steps=2,
        tool_calls=[],
        output=MockPersistenceResponse(tool_result="", other_field="step2"),
    )

    llm_client.chat_completions.side_effect = [
        (step1, {"stats": "dummy"}),
        (step2, {"stats": "dummy"}),
    ]

    result = await agent.run(task="multiple outputs", max_iterations=2)

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
        response_model=MockPersistenceResponse,
        auto_persist=False,
    )

    StepOutput = get_step_output_type(MockPersistenceResponse)

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
        output=MockPersistenceResponse(tool_result="original", other_field="from_llm"),
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


@pytest.mark.asyncio
async def test_planning_agent_tool_error_logging(caplog):
    # Setup mocks
    kernel = MagicMock(spec=FunctionKernel)
    kernel.get_tool_descriptions = AsyncMock(return_value="Mock Tool Description")

    # Simulate a tool failure
    kernel.call_tool = AsyncMock(
        side_effect=Exception(
            "Python tool 'reorder' argument validation failed: 2 validation errors for reorder_input\nhotel_search_results\n  Field required [type=missing, input_value={}, input_type=dict]\npositions\n  Field required [type=missing, input_value={}, input_type=dict]"
        )
    )

    run_context = MagicMock(spec=RunContext)
    llm_client = MagicMock(spec=LLMClient)

    agent = PlanningAgent(
        kernel=kernel,
        run_context=run_context,
        llm_client=llm_client,
        input_data={},
        response_model=MockErrorResponse,
    )

    StepOutput = get_step_output_type(MockErrorResponse)

    step1 = StepOutput(
        short_explanation="Calling tool",
        long_explanation="I will call the tool",
        step_number=0,
        max_steps=1,
        tool_calls=[ToolCall(name="python://reorder", args="{}", call_id="c1")],
        output=MockErrorResponse(result="done"),
    )

    llm_client.chat_completions.side_effect = [
        (step1, {"stats": "dummy"}),
    ]

    # Set log level to capture warnings/errors
    with caplog.at_level(logging.WARNING):
        await agent.run(task="test", max_iterations=1)

    error_logs = [
        record
        for record in caplog.records
        if record.levelname == "ERROR"
        and "Tool python://reorder failed" in record.message
    ]
    warning_logs = [
        record
        for record in caplog.records
        if record.levelname == "WARNING"
        and "Tool python://reorder failed" in record.message
    ]

    assert len(error_logs) > 0
    assert len(warning_logs) == 0


@pytest.mark.asyncio
async def test_planning_agent_tool_args_parse_error_logging(caplog):
    # Setup mocks
    kernel = MagicMock(spec=FunctionKernel)
    kernel.get_tool_descriptions = AsyncMock(return_value="Mock Tool Description")

    run_context = MagicMock(spec=RunContext)
    llm_client = MagicMock(spec=LLMClient)

    agent = PlanningAgent(
        kernel=kernel,
        run_context=run_context,
        llm_client=llm_client,
        input_data={},
        response_model=MockErrorResponse,
    )

    StepOutput = get_step_output_type(MockErrorResponse)

    # Tool call with invalid JSON in args string
    step1 = StepOutput(
        short_explanation="Calling tool",
        long_explanation="I will call the tool with invalid args",
        step_number=0,
        max_steps=1,
        tool_calls=[
            ToolCall(name="python://test", args='{"invalid": json}', call_id="c1")
        ],
        output=MockErrorResponse(result="done"),
    )

    llm_client.chat_completions.side_effect = [
        (step1, {"stats": "dummy"}),
    ]

    # Set log level
    with caplog.at_level(logging.WARNING):
        await agent.run(task="test", max_iterations=1)

    error_logs = [
        record
        for record in caplog.records
        if record.levelname == "ERROR" and "Failed to parse tool args" in record.message
    ]

    assert len(error_logs) > 0


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
        tool_calls=[ToolCall(name="python://fail", call_id="c1", args="{}")],
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

    llm_client.chat_completions.side_effect = [(step1, {}), (step2, {1: 1})]

    # Run agent
    result = await agent.run(task="task", max_iterations=2)

    # Assertions
    assert result == final_output
    assert kernel.call_tool.called

    # Check if the error message was put into planner_context
    assert "c1" in agent._planner_context
    assert "Error: Validation error: field required" in agent._planner_context["c1"]


@pytest.mark.asyncio
async def test_planning_agent_real_python_tool_gcd():
    # Setup real FunctionKernel
    kernel = FunctionKernel()

    @pythontool
    def compute_gcd(a: int, b: int) -> int:
        """Compute the greatest common divisor of two integers."""
        import math

        return math.gcd(a, b)

    kernel.register_python_tool("gcd_tool", compute_gcd)

    run_context = MagicMock(spec=RunContext)
    llm_client = MagicMock(spec=LLMClient)

    input_data = {"val1": 48, "val2": 18}
    task = "Compute GCD of 48 and 18"

    agent = PlanningAgent(
        kernel=kernel,
        run_context=run_context,
        llm_client=llm_client,
        input_data=input_data,
        response_model=MockResponse,
    )

    StepOutput = get_step_output_type(MockResponse)

    # First iteration: Call GCD tool
    step1 = StepOutput(
        short_explanation="Calculating GCD",
        long_explanation="I will use the gcd_tool to compute the GCD of 48 and 18.",
        step_number=0,
        max_steps=2,
        tool_calls=[
            ToolCall(
                name="python://gcd_tool",
                call_id="gcd_result",
                args='{"a": 48, "b": 18}',
            )
        ],
        output=None,
    )

    # Second iteration: Final result (GCD of 48 and 18 is 6)
    step2 = StepOutput(
        short_explanation="GCD computed",
        long_explanation="The GCD of 48 and 18 is 6.",
        step_number=1,
        max_steps=2,
        tool_calls=[],
        output=MockResponse(answer="The GCD is 6"),
    )

    llm_client.chat_completions.side_effect = [
        (step1, {"stats": "step1"}),
        (step2, {"stats": "step2"}),
    ]

    # Run agent
    result = await agent.run(task=task, max_iterations=5)

    # Assertions
    assert isinstance(result, MockResponse)
    assert result.answer == "The GCD is 6"
    # FunctionKernel wraps primitive return values in a model with a 'result' field
    assert agent._planner_context["gcd_result"].result == 6
    assert llm_client.chat_completions.call_count == 2
