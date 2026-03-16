import pytest
import logging
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel
from kavalai.agents.planning_agent import PlanningAgent, get_step_output_type, ToolCall
from kavalai.functionkernel import FunctionKernel
from kavalai.agents.run_context import RunContext
from kavalai.llm_clients.llm_client import LLMClient


class MockResponse(BaseModel):
    result: str


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
        response_model=MockResponse,
    )

    StepOutput = get_step_output_type(MockResponse)

    step1 = StepOutput(
        short_explanation="Calling tool",
        long_explanation="I will call the tool",
        step_number=0,
        max_steps=1,
        tool_calls=[ToolCall(name="python://reorder", args="{}", call_id="c1")],
        output=MockResponse(result="done"),
    )

    llm_client.chat_completions.side_effect = [
        (step1, {"stats": "dummy"}),
    ]

    # Set log level to capture warnings/errors
    with caplog.at_level(logging.WARNING):
        await agent.run(task="test", max_iterations=1)

    # Check if the error was logged
    # Current behavior: it should be in caplog.records as WARNING
    # Target behavior: it should be in caplog.records as ERROR

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
    print("\nVerified: Tool failure logged as ERROR")


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
        response_model=MockResponse,
    )

    StepOutput = get_step_output_type(MockResponse)

    # Tool call with invalid JSON in args string
    step1 = StepOutput(
        short_explanation="Calling tool",
        long_explanation="I will call the tool with invalid args",
        step_number=0,
        max_steps=1,
        tool_calls=[
            ToolCall(name="python://test", args='{"invalid": json}', call_id="c1")
        ],
        output=MockResponse(result="done"),
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
    print("\nVerified: Tool args parse failure logged as ERROR")
