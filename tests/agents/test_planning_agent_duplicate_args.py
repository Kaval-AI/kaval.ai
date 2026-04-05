import logging
import pytest
import json
from unittest.mock import MagicMock
from pydantic import BaseModel

from kavalai.agents.planning_agent import PlanningAgent, ToolCall, get_step_output_type
from kavalai.functionkernel import FunctionKernel, pythontool
from kavalai.agents.run_context import RunContext
from kavalai.llm_clients.llm_client import LLMClient


class MockResponse(BaseModel):
    answer: str


@pytest.mark.asyncio
async def test_planning_agent_duplicate_args_merge_with_precedence(caplog):
    """Test that duplicate arguments are merged with literal_args > planner_context_args > input_args precedence."""
    kernel = FunctionKernel()

    @pythontool
    def test_tool(val: str, other: str) -> str:
        return f"{val}:{other}"

    kernel.register_python_tool("test_tool", test_tool)

    run_context = RunContext()
    llm_client = MagicMock(spec=LLMClient)

    # Setup input data and planner context
    input_data = {"val_input": "from_input", "other_input": "other_input"}
    agent = PlanningAgent(
        kernel=kernel,
        run_context=run_context,
        llm_client=llm_client,
        input_data=input_data,
        response_model=MockResponse,
    )
    agent._planner_context["val_context"] = "from_context"

    StepOutput = get_step_output_type(MockResponse)

    # Create a tool call with duplicate 'val' argument across all three sources
    # Expected precedence: literal_args > planner_context_args > input_args
    step1 = StepOutput(
        short_explanation="Testing merge",
        instructions="Duplicate args test",
        tool_calls=[
            ToolCall(
                name="python://test_tool",
                call_id="call1",
                literal_args=json.dumps({"val": "from_literal"}),
                planner_context_args=json.dumps({"val": "val_context"}),
                input_args=json.dumps({"val": "val_input", "other": "other_input"}),
            )
        ],
        output=MockResponse(answer="Done"),
    )

    llm_client.chat_completions.return_value = (step1, {})

    # Set log level to capture error logs
    with caplog.at_level(logging.ERROR):
        await agent.run(task_name="test_task", task="task", max_iterations=1)

    # Check that literal_args took precedence
    assert agent._planner_context["call1"].result == "from_literal:other_input"

    # Check that error was logged
    error_logs = [
        record.message
        for record in caplog.records
        if record.levelname == "ERROR"
        and "Duplicate argument names found in ToolCall: {'val'}" in record.message
    ]
    assert len(error_logs) > 0
    assert (
        "precedence: literal_args > planner_context_args > input_args" in error_logs[0]
    )


@pytest.mark.asyncio
async def test_planning_agent_duplicate_args_planner_context_over_input(caplog):
    """Test that planner_context_args takes precedence over input_args when literal_args is not specified."""
    kernel = FunctionKernel()

    @pythontool
    def test_tool(val: str) -> str:
        return val

    kernel.register_python_tool("test_tool", test_tool)

    run_context = RunContext()
    llm_client = MagicMock(spec=LLMClient)

    # Setup input data and planner context
    input_data = {"val_input": "from_input"}
    agent = PlanningAgent(
        kernel=kernel,
        run_context=run_context,
        llm_client=llm_client,
        input_data=input_data,
        response_model=MockResponse,
    )
    agent._planner_context["val_context"] = "from_context"

    StepOutput = get_step_output_type(MockResponse)

    # Create a tool call with duplicate 'val' in planner_context_args and input_args only
    step1 = StepOutput(
        short_explanation="Testing merge",
        instructions="Context over input test",
        tool_calls=[
            ToolCall(
                name="python://test_tool",
                call_id="call1",
                planner_context_args=json.dumps({"val": "val_context"}),
                input_args=json.dumps({"val": "val_input"}),
            )
        ],
        output=MockResponse(answer="Done"),
    )

    llm_client.chat_completions.return_value = (step1, {})

    with caplog.at_level(logging.ERROR):
        await agent.run(task_name="test_task", task="task", max_iterations=1)

    # Check that planner_context_args took precedence over input_args
    assert agent._planner_context["call1"].result == "from_context"

    # Check that error was logged
    error_logs = [
        record.message
        for record in caplog.records
        if record.levelname == "ERROR" and "Duplicate argument names" in record.message
    ]
    assert len(error_logs) > 0
