import logging
import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel, Field
from kavalai.agents.planning_agent import PlanningAgent, ToolCall, get_step_output_type
from kavalai.agents.task_logger import TaskLogger
from kavalai.functionkernel import FunctionKernel, pythontool
from kavalai.agents.run_context import RunContext
from kavalai.llm_clients.llm_client import LLMClient
from kavalai.llm_clients.common import Streamer
from kavalai.agents.agent_service import AgentService


@pytest.fixture
def session_maker(agents_session_maker):
    return agents_session_maker


@pytest.fixture
def agent_service(session_maker):
    return AgentService(session_maker)


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
    kernel.get_tool_descriptions = AsyncMock(return_value="[]")
    kernel.call_tool = AsyncMock(return_value="Tool Result")

    run_context = RunContext()
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
        agent_service=run_context.agent_service,
    )

    StepOutput = get_step_output_type(MockResponse)

    # Mock LLM response for first iteration (tool call)
    step1 = StepOutput(
        short_explanation="Step 1",
        instructions="Planning to call a tool",
        tool_calls=[
            ToolCall(
                name="python://tool",
                call_id="call1",
                literal_args=json.dumps({"a": 1}),
            )
        ],
        output=None,
    )

    # Mock LLM response for second iteration (final answer)
    step2 = StepOutput(
        short_explanation="Step 2",
        instructions="Finished",
        tool_calls=[],
        output=MockResponse(answer="Final Answer"),
    )

    llm_client.chat_completions.side_effect = [
        (step1, {"stats": "dummy"}),
        (step2, {"stats": "dummy"}),
    ]

    # Run agent
    result = await agent.run(task_name="test_task", task=task, max_iterations=5)

    # Assertions
    assert isinstance(result, MockResponse)
    assert result.answer == "Final Answer"
    assert len(agent._step_outputs) == 2
    assert agent._planner_context["call1"] == "Tool Result"

    assert llm_client.chat_completions.call_count == 2


@pytest.mark.asyncio
async def test_planning_agent_resolves_planner_context_references():
    # Setup real FunctionKernel with a tool
    kernel = FunctionKernel()

    class NestedModel(BaseModel):
        val: str

    @pythontool
    def test_tool(data: NestedModel) -> str:
        return f"Resolved: {data.val}"

    kernel.register_python_tool("test_tool", test_tool)

    run_context = RunContext()
    llm_client = MagicMock(spec=LLMClient)

    agent = PlanningAgent(
        kernel=kernel,
        run_context=run_context,
        llm_client=llm_client,
        input_data={},
        response_model=MockResponse,
    )

    StepOutput = get_step_output_type(MockResponse)

    # Injected result in planner_context
    nested_result = NestedModel(val="target")
    agent._planner_context["prev_call"] = nested_result

    # Step calling tool with reference to 'prev_call'
    step1 = StepOutput(
        short_explanation="Calling tool",
        instructions="Using reference",
        tool_calls=[
            ToolCall(
                name="python://test_tool",
                call_id="curr_call",
                planner_context_args=json.dumps({"data": "prev_call"}),
            )
        ],
        output=MockResponse(answer="Done"),
    )

    llm_client.chat_completions.return_value = (step1, {})

    await agent.run(task_name="test_task", task="task", max_iterations=1)

    # FunctionKernel wraps primitive return values in a model with a 'result' field
    assert agent._planner_context["curr_call"].result == "Resolved: target"


@pytest.mark.asyncio
async def test_planning_agent_resolves_args_from_new_fields():
    # Setup real FunctionKernel with a tool
    kernel = FunctionKernel()

    @pythontool
    def test_tool(literal_val: str, input_val: str, context_val: str) -> str:
        return f"Literal: {literal_val}, Input: {input_val}, Context: {context_val}"

    kernel.register_python_tool("test_tool", test_tool)

    run_context = RunContext()
    llm_client = MagicMock(spec=LLMClient)

    input_data = {"user_name": "Alice"}
    agent = PlanningAgent(
        kernel=kernel,
        run_context=run_context,
        llm_client=llm_client,
        input_data=input_data,
        response_model=MockResponse,
    )

    StepOutput = get_step_output_type(MockResponse)

    # Injected result in planner_context
    agent._planner_context["prev_result"] = "Previous Result"

    # Step calling tool with various argument sources
    step1 = StepOutput(
        short_explanation="Testing new fields",
        instructions="Using new literal_args, input_args and planner_context_args fields",
        tool_calls=[
            ToolCall(
                name="python://test_tool",
                call_id="curr_call",
                literal_args=json.dumps({"literal_val": "Literal Value"}),
                input_args=json.dumps({"input_val": "user_name"}),
                planner_context_args=json.dumps({"context_val": "prev_result"}),
            )
        ],
        output=MockResponse(answer="Done"),
    )

    llm_client.chat_completions.return_value = (step1, {})

    await agent.run(task_name="test_task", task="task", max_iterations=1)

    assert (
        agent._planner_context["curr_call"].result
        == "Literal: Literal Value, Input: Alice, Context: Previous Result"
    )


@pytest.mark.asyncio
async def test_planning_agent_resolves_args_priority():
    kernel = FunctionKernel()

    @pythontool
    def test_tool(val: str) -> str:
        return val

    kernel.register_python_tool("test_tool", test_tool)

    run_context = RunContext()
    llm_client = MagicMock(spec=LLMClient)

    # All sources have 'val'
    input_data = {"val_key": "input"}
    agent = PlanningAgent(
        kernel=kernel,
        run_context=run_context,
        llm_client=llm_client,
        input_data=input_data,
        response_model=MockResponse,
    )
    agent._planner_context["val_key"] = "context"

    StepOutput = get_step_output_type(MockResponse)

    # Test literal
    step1 = StepOutput(
        short_explanation="Testing priority",
        instructions="Literal",
        tool_calls=[
            ToolCall(
                name="python://test_tool",
                call_id="call1",
                literal_args=json.dumps({"val": "literal"}),
            )
        ],
        output=MockResponse(answer="Done"),
    )

    llm_client.chat_completions.return_value = (step1, {})
    await agent.run(task_name="test_task", task="task", max_iterations=1)
    assert agent._planner_context["call1"].result == "literal"

    # Test context resolution
    step2 = StepOutput(
        short_explanation="Testing priority",
        instructions="Context resolution",
        tool_calls=[
            ToolCall(
                name="python://test_tool",
                call_id="call2",
                planner_context_args=json.dumps({"val": "val_key"}),
            )
        ],
        output=MockResponse(answer="Done"),
    )

    llm_client.chat_completions.side_effect = [(step2, {})]
    await agent.run(task_name="test_task", task="task", max_iterations=1)
    assert agent._planner_context["call2"].result == "context"


@pytest.mark.asyncio
async def test_planning_agent_resolves_nested_args_from_planner_context():
    # Setup real FunctionKernel with a tool
    kernel = FunctionKernel()

    class NestedModel(BaseModel):
        val: str

    @pythontool
    def test_tool(data: NestedModel) -> str:
        return f"Resolved: {data.val}"

    kernel.register_python_tool("test_tool", test_tool)

    run_context = RunContext()
    llm_client = MagicMock(spec=LLMClient)

    agent = PlanningAgent(
        kernel=kernel,
        run_context=run_context,
        llm_client=llm_client,
        input_data={},
        response_model=MockResponse,
    )

    StepOutput = get_step_output_type(MockResponse)

    # Injected result in planner_context
    nested_result = NestedModel(val="target")
    agent._planner_context["prev_call"] = nested_result

    # Step calling tool with reference to 'prev_call' via template string
    step1 = StepOutput(
        short_explanation="Calling tool",
        instructions="Using reference",
        tool_calls=[
            ToolCall(
                name="python://test_tool",
                call_id="curr_call",
                literal_args=json.dumps({"data": "{{context.prev_call}}"}),
            )
        ],
        output=MockResponse(answer="Done"),
    )

    llm_client.chat_completions.return_value = (step1, {})

    await agent.run(task_name="test_task", task="task", max_iterations=1)

    assert agent._planner_context["curr_call"].result == "Resolved: target"


@pytest.mark.asyncio
async def test_planning_agent_max_iterations():
    kernel = MagicMock(spec=FunctionKernel)
    kernel.get_tool_descriptions = AsyncMock(return_value="[]")
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
        instructions="Waiting...",
        tool_calls=[],
        output=None,
    )

    llm_client.chat_completions.return_value = (step, {})

    result = await agent.run(task_name="test_task", task="task", max_iterations=3)

    assert result is None
    assert llm_client.chat_completions.call_count == 3


@pytest.mark.asyncio
async def test_planning_agent_streaming():
    # Setup mocks
    kernel = MagicMock(spec=FunctionKernel)
    kernel.get_tool_descriptions = AsyncMock(return_value="[]")

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
        instructions="Finished the task",
        tool_calls=[],
        output=final_output,
    )

    llm_client.chat_completions.return_value = (step, {"stats": "dummy"})

    # Run agent
    result = await agent.run(task_name="test_task", task=task, max_iterations=1)

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
async def test_planning_agent_persist_to():
    # Setup mocks
    kernel = MagicMock(spec=FunctionKernel)
    kernel.get_tool_descriptions = AsyncMock(return_value="[]")
    kernel.call_tool = AsyncMock(return_value="PERSISTED_VALUE")

    run_context = RunContext(data={})
    llm_client = MagicMock(spec=LLMClient)

    input_data = {}
    task = "Use tool and persist result"

    agent = PlanningAgent(
        kernel=kernel,
        run_context=run_context,
        llm_client=llm_client,
        input_data=input_data,
        response_model=MockPersistenceResponse,
    )

    StepOutput = get_step_output_type(MockPersistenceResponse)

    # First iteration: call tool with persist_to="my_key"
    step1 = StepOutput(
        short_explanation="Calling tool",
        instructions="I will call the tool to get some data and persist it",
        tool_calls=[
            ToolCall(
                name="python://test_tool",
                call_id="c1",
                literal_args="{}",
                persist_to="my_key",
            )
        ],
        output=None,
    )

    # Second iteration: return output
    step2 = StepOutput(
        short_explanation="Returning result",
        instructions="I am returning the result",
        tool_calls=[],
        output=MockPersistenceResponse(tool_result="done", other_field="ok"),
    )

    llm_client.chat_completions.side_effect = [
        (step1, {"stats": "dummy"}),
        (step2, {"stats": "dummy"}),
    ]

    # Run agent
    result = await agent.run(task_name="test_task", task=task, max_iterations=2)

    # Assertions
    assert result is not None
    assert result.tool_result == "done"
    # Check if value was persisted to run_context.data
    assert run_context.data.get("my_key") == "PERSISTED_VALUE"
    assert "my_key" in run_context.data, "Value should be persisted to run_context.data"


@pytest.mark.asyncio
async def test_planning_agent_tool_error_logging(caplog):
    # Setup mocks
    kernel = MagicMock(spec=FunctionKernel)
    kernel.get_tool_descriptions = AsyncMock(return_value="[]")

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
        instructions="I will call the tool",
        tool_calls=[ToolCall(name="python://reorder", literal_args="{}", call_id="c1")],
        output=MockErrorResponse(result="done"),
    )

    llm_client.chat_completions.side_effect = [
        (step1, {"stats": "dummy"}),
    ]

    # Set log level to capture warnings/errors
    with caplog.at_level(logging.WARNING):
        await agent.run(task_name="test_task", task="test", max_iterations=1)

    error_logs = [
        record.message
        for record in caplog.records
        if record.levelname == "ERROR"
        and "Tool python://reorder failed" in record.message
    ]
    warning_logs = [
        record.message
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
    kernel.get_tool_descriptions = AsyncMock(return_value="[]")

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

    # Simulate a tool failure
    kernel.call_tool = AsyncMock(side_effect=Exception("Tool failed"))

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
        instructions="I will call the tool",
        tool_calls=[
            ToolCall(
                name="python://test",
                literal_args='{"invalid": "json"}',
                call_id="c1",
            )
        ],
        output=MockErrorResponse(result="done"),
    )

    llm_client.chat_completions.side_effect = [
        (step1, {"stats": "dummy"}),
    ]

    # Set log level
    with caplog.at_level(logging.ERROR):
        await agent.run(task_name="test_task", task="test", max_iterations=1)

    error_logs = [
        record.message
        for record in caplog.records
        if record.levelname == "ERROR" and "Tool python://test failed" in record.message
    ]

    assert len(error_logs) > 0


@pytest.mark.asyncio
async def test_planning_agent_tool_failure_handling():
    # Setup mocks
    kernel = MagicMock(spec=FunctionKernel)
    kernel.get_tool_descriptions = AsyncMock(return_value="[]")

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
        instructions="Need to call a tool",
        tool_calls=[ToolCall(name="python://fail", call_id="c1", literal_args="{}")],
        output=None,
    )

    # Second step: LLM provides final output based on (failed) tool result
    final_output = MockResponse(answer="Handled failure")
    step2 = StepOutput(
        short_explanation="Done",
        instructions="Finished the task",
        tool_calls=[],
        output=final_output,
    )

    llm_client.chat_completions.side_effect = [(step1, {}), (step2, {1: 1})]

    # Run agent
    result = await agent.run(task_name="test_task", task="task", max_iterations=2)

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

    run_context = RunContext()
    run_context.agent_service = AsyncMock()
    llm_client = MagicMock(spec=LLMClient)

    input_data = {"val1": 48, "val2": 18}
    task = "Compute GCD of 48 and 18"

    agent = PlanningAgent(
        kernel=kernel,
        run_context=run_context,
        llm_client=llm_client,
        input_data=input_data,
        response_model=MockResponse,
        agent_service=run_context.agent_service,
    )

    StepOutput = get_step_output_type(MockResponse)

    # First iteration: Call GCD tool
    step1 = StepOutput(
        short_explanation="Calculating GCD",
        instructions="I will use the gcd_tool to compute the GCD of 48 and 18.",
        tool_calls=[
            ToolCall(
                name="python://gcd_tool",
                call_id="gcd_result",
                literal_args=json.dumps({"a": 48, "b": 18}),
            )
        ],
        output=None,
    )

    # Second iteration: Final result (GCD of 48 and 18 is 6)
    step2 = StepOutput(
        short_explanation="GCD computed",
        instructions="The GCD of 48 and 18 is 6.",
        tool_calls=[],
        output=MockResponse(answer="The GCD is 6"),
    )

    llm_client.chat_completions.side_effect = [
        (step1, {"stats": "step1"}),
        (step2, {"stats": "step2"}),
    ]

    # Run agent
    result = await agent.run(task_name="test_task", task=task, max_iterations=5)

    # Assertions
    assert isinstance(result, MockResponse)
    assert result.answer == "The GCD is 6"
    # FunctionKernel wraps primitive return values in a model with a 'result' field
    assert agent._planner_context["gcd_result"].result == 6
    assert llm_client.chat_completions.call_count == 2


@pytest.mark.asyncio
async def test_planning_agent_complex_nested_models():
    # Setup real FunctionKernel
    kernel = FunctionKernel()

    class NestedInput(BaseModel):
        field_a: str
        field_b: int

    class ComplexInput(BaseModel):
        name: str
        nested: NestedInput

    class NestedOutput(BaseModel):
        success: bool
        message: str

    class ComplexOutput(BaseModel):
        code: int
        data: NestedOutput

    @pythontool
    def complex_tool(name: str, nested: NestedInput) -> ComplexOutput:
        """A tool with nested pydantic models."""
        return ComplexOutput(
            code=200,
            data=NestedOutput(
                success=True,
                message=f"Hello {name}, you sent {nested.field_a} and {nested.field_b}",
            ),
        )

    kernel.register_python_tool("complex_tool", complex_tool)

    run_context = MagicMock(spec=RunContext)
    llm_client = MagicMock(spec=LLMClient)

    task = "Use the complex tool"

    agent = PlanningAgent(
        kernel=kernel,
        run_context=run_context,
        llm_client=llm_client,
        input_data={},
        response_model=MockResponse,
    )

    StepOutput = get_step_output_type(MockResponse)

    # First iteration: Call complex tool
    tool_args = {"name": "Junie", "nested": {"field_a": "test", "field_b": 42}}
    step1 = StepOutput(
        short_explanation="Calling complex tool",
        instructions="I will call the complex tool with nested data.",
        tool_calls=[
            ToolCall(
                name="python://complex_tool",
                call_id="complex_result",
                literal_args=json.dumps(tool_args),
            )
        ],
        output=None,
    )

    # Second iteration: Final result
    step2 = StepOutput(
        short_explanation="Done",
        instructions="The complex tool returned a nested response.",
        tool_calls=[],
        output=MockResponse(answer="Success"),
    )

    llm_client.chat_completions.side_effect = [
        (step1, {"stats": "step1"}),
        (step2, {"stats": "step2"}),
    ]

    # Run agent
    result = await agent.run(task_name="test_task", task=task, max_iterations=5)

    # Assertions
    assert isinstance(result, MockResponse)
    assert result.answer == "Success"

    # Verify tool result in planner context
    tool_result = agent._planner_context["complex_result"]
    assert isinstance(tool_result, ComplexOutput)
    assert tool_result.code == 200
    assert tool_result.data.success is True
    assert "Junie" in tool_result.data.message
    assert "test" in tool_result.data.message
    assert "42" in tool_result.data.message

    assert llm_client.chat_completions.call_count == 2


@pytest.mark.asyncio
async def test_planning_agent_logs_agent_task(agent_service, session_maker):
    """Test that PlanningAgent logs the overall agent task with system prompt."""
    # Create agent, session, and run
    agent_db = await agent_service.get_or_create_agent(
        name="test_agent",
        description="Test agent",
        input_schema={},
        output_schema={},
        workflow={},
    )
    session = await agent_service.get_or_create_session(agent_id=agent_db.id)
    run = await agent_service.create_run(session_id=session.id, input_data={})

    # Setup mocks
    kernel = MagicMock(spec=FunctionKernel)
    kernel.get_tool_descriptions = AsyncMock(return_value="[Tool: test_tool]")

    run_context = RunContext(agent_service=agent_service)
    run_context.agent_id = agent_db.id
    run_context.session_id = session.id
    run_context.run_id = run.id

    llm_client = MagicMock(spec=LLMClient)

    input_data = {"query": "test query"}
    task = "Solve this task"

    # Create task logger
    task_logger = TaskLogger(agent_service, run_context)

    # Create planning agent with task_logger
    planning_agent = PlanningAgent(
        kernel=kernel,
        run_context=run_context,
        llm_client=llm_client,
        input_data=input_data,
        response_model=MockResponse,
        agent_service=agent_service,
        task_logger=task_logger,
    )

    StepOutput = get_step_output_type(MockResponse)

    # Mock LLM response - single step that completes
    final_output = MockResponse(answer="Final Answer")
    step = StepOutput(
        short_explanation="Done",
        instructions="Completed the task",
        tool_calls=[],
        output=final_output,
    )

    llm_client.chat_completions.return_value = (step, {"stats": "dummy"})

    # Run agent
    result = await planning_agent.run(
        task_name="planning_task", task=task, max_iterations=5
    )

    # Verify result
    assert result == final_output

    # Verify the agent task was logged with system prompt
    async with session_maker() as db_session:
        from kavalai.agents.db import Task
        from sqlalchemy import select

        result = await db_session.execute(select(Task).where(Task.run_id == run.id))
        tasks = result.scalars().all()

        assert len(tasks) == 2
        # First task: step 0
        step_task = tasks[0]
        assert step_task.name == "planning_task_step_0"

        # Second task: overall agent task
        task_db = tasks[1]
        assert task_db.name == "planning_task"
        assert task_db.prompt is not None
        assert "You are a planning agent" in task_db.prompt
        assert "Solve this task" in task_db.prompt
        assert "[Tool: test_tool]" in task_db.prompt
        assert task_db.inputs == {"query": "test query"}
        assert task_db.output == {"answer": "Final Answer"}
        assert task_db.duration_seconds > 0


@pytest.mark.asyncio
async def test_planning_agent_logs_tool_calls(agent_service, session_maker):
    """Test that PlanningAgent logs individual tool calls."""
    # Create agent, session, and run
    agent_db = await agent_service.get_or_create_agent(
        name="test_agent",
        description="Test agent",
        input_schema={},
        output_schema={},
        workflow={},
    )
    session = await agent_service.get_or_create_session(agent_id=agent_db.id)
    run = await agent_service.create_run(session_id=session.id, input_data={})

    # Setup real kernel with a tool
    kernel = FunctionKernel()

    @pythontool
    def test_tool(query: str) -> str:
        return f"Result for {query}"

    kernel.register_python_tool("test_tool", test_tool)

    run_context = RunContext(agent_service=agent_service)
    run_context.agent_id = agent_db.id
    run_context.session_id = session.id
    run_context.run_id = run.id

    llm_client = MagicMock(spec=LLMClient)

    input_data = {"query": "test query"}
    task = "Use the tool"

    # Create task logger
    task_logger = TaskLogger(agent_service, run_context)

    # Create planning agent with task_logger
    planning_agent = PlanningAgent(
        kernel=kernel,
        run_context=run_context,
        llm_client=llm_client,
        input_data=input_data,
        response_model=MockResponse,
        agent_service=agent_service,
        task_logger=task_logger,
    )

    StepOutput = get_step_output_type(MockResponse)

    # First step: call tool
    step1 = StepOutput(
        short_explanation="Calling tool",
        instructions="Using test_tool",
        tool_calls=[
            ToolCall(
                name="python://test_tool",
                call_id="call1",
                literal_args=json.dumps({"query": "search term"}),
            )
        ],
        output=None,
    )

    # Second step: return result
    step2 = StepOutput(
        short_explanation="Done",
        instructions="Completed",
        tool_calls=[],
        output=MockResponse(answer="Final Answer"),
    )

    llm_client.chat_completions.side_effect = [
        (step1, {"stats": "dummy"}),
        (step2, {"stats": "dummy"}),
    ]

    # Run agent
    result = await planning_agent.run(
        task_name="planning_task", task=task, max_iterations=5
    )

    # Verify result
    assert result.answer == "Final Answer"

    # Verify both the agent task and tool call were logged
    async with session_maker() as db_session:
        from kavalai.agents.db import Task
        from sqlalchemy import select

        result = await db_session.execute(
            select(Task).where(Task.run_id == run.id).order_by(Task.created_at)
        )
        tasks = result.scalars().all()

        # Should have 4 tasks: tool call, step_0, step_1, agent task
        assert len(tasks) == 4

        # First task: tool call (from step 0 processing)
        tool_task = tasks[0]
        assert tool_task.name == "python://test_tool"
        assert tool_task.inputs == {"arguments": {"query": "search term"}}
        assert "Result for search term" in str(tool_task.output)

        # Second task: step 0
        step0_task = tasks[1]
        assert step0_task.name == "planning_task_step_0"

        # Third task: step 1
        step1_task = tasks[2]
        assert step1_task.name == "planning_task_step_1"

        # Fourth task: overall agent task
        agent_task = tasks[3]
        assert agent_task.name == "planning_task"


@pytest.mark.asyncio
async def test_planning_agent_generates_deterministic_call_id():
    """Test that PlanningAgent generates call_id for tool calls if missing from LLM response."""
    from kavalai.agents.planning_agent import ToolCall, get_step_output_type

    kernel = MagicMock(spec=FunctionKernel)
    kernel.get_tool_descriptions = AsyncMock(return_value="[]")
    kernel.call_tool = AsyncMock(return_value="Tool Result")

    run_context = RunContext()
    llm_client = MagicMock(spec=LLMClient)

    agent = PlanningAgent(
        kernel=kernel,
        run_context=run_context,
        llm_client=llm_client,
        input_data={},
        response_model=MockResponse,
    )

    StepOutput = get_step_output_type(MockResponse)

    # Mock LLM response with missing call_id
    step0 = StepOutput(
        short_explanation="Step 0",
        instructions="Planning to call a tool",
        tool_calls=[
            ToolCall(
                name="python://tool1",
                call_id=None,
                literal_args=json.dumps({"a": 1}),
            ),
            ToolCall(
                name="python://tool2",
                call_id="provided_id",
                literal_args=json.dumps({"b": 2}),
            ),
        ],
        output=None,
    )

    # Final response
    step1 = StepOutput(
        short_explanation="Step 1",
        instructions="Finished",
        tool_calls=[],
        output=MockResponse(answer="Final Answer"),
    )

    llm_client.chat_completions.side_effect = [
        (step0, {"stats": "dummy"}),
        (step1, {"stats": "dummy"}),
    ]

    # Run agent
    await agent.run(task_name="test_task", task="task", max_iterations=2)

    # Check that call_id was generated deterministically for the first tool call
    tool_calls = agent._step_outputs[0].tool_calls
    assert tool_calls[0].call_id == "tool_call_0_0"
    assert tool_calls[1].call_id == "provided_id"

    # Check it is in planner_context
    assert "tool_call_0_0" in agent._planner_context
    assert agent._planner_context["tool_call_0_0"] == "Tool Result"
    assert "provided_id" in agent._planner_context


@pytest.mark.asyncio
async def test_planning_agent_logging_without_agent_service():
    """Test that PlanningAgent works without agent_service (no logging)."""
    # Setup mocks
    kernel = MagicMock(spec=FunctionKernel)
    kernel.get_tool_descriptions = AsyncMock(return_value="[]")
    kernel.call_tool = AsyncMock(return_value="Tool Result")

    run_context = RunContext(agent_service=None)
    llm_client = MagicMock(spec=LLMClient)

    input_data = {"key": "value"}
    task = "Solve this task"

    # Create planning agent without task_logger (None agent_service)
    planning_agent = PlanningAgent(
        kernel=kernel,
        run_context=run_context,
        llm_client=llm_client,
        input_data=input_data,
        response_model=MockResponse,
        agent_service=None,
        task_logger=None,
    )

    StepOutput = get_step_output_type(MockResponse)

    # Mock LLM response
    final_output = MockResponse(answer="Final Answer")
    step = StepOutput(
        short_explanation="Done",
        instructions="Finished",
        tool_calls=[],
        output=final_output,
    )

    llm_client.chat_completions.return_value = (step, {"stats": "dummy"})

    # Run agent - should not raise any errors
    result = await planning_agent.run(
        task_name="test_task", task=task, max_iterations=1
    )

    # Verify result
    assert result == final_output


@pytest.mark.asyncio
async def test_planning_agent_logs_tool_call_errors(agent_service, session_maker):
    """Test that PlanningAgent logs tool call errors."""
    # Create agent, session, and run
    agent_db = await agent_service.get_or_create_agent(
        name="test_agent",
        description="Test agent",
        input_schema={},
        output_schema={},
        workflow={},
    )
    session = await agent_service.get_or_create_session(agent_id=agent_db.id)
    run = await agent_service.create_run(session_id=session.id, input_data={})

    # Setup kernel that will fail
    kernel = MagicMock(spec=FunctionKernel)
    kernel.get_tool_descriptions = AsyncMock(return_value="[Tool: failing_tool]")
    kernel.call_tool = AsyncMock(
        side_effect=Exception("ValidationError: Missing required field")
    )

    run_context = RunContext(agent_service=agent_service)
    run_context.agent_id = agent_db.id
    run_context.session_id = session.id
    run_context.run_id = run.id

    llm_client = MagicMock(spec=LLMClient)

    input_data = {"query": "test"}
    task = "Call the tool"

    # Create task logger
    from kavalai.agents.task_logger import TaskLogger

    task_logger = TaskLogger(agent_service, run_context)

    # Create planning agent with task_logger
    planning_agent = PlanningAgent(
        kernel=kernel,
        run_context=run_context,
        llm_client=llm_client,
        input_data=input_data,
        response_model=MockResponse,
        agent_service=agent_service,
        task_logger=task_logger,
    )

    StepOutput = get_step_output_type(MockResponse)

    # Step 1: call tool that will fail
    step1 = StepOutput(
        short_explanation="Calling tool",
        instructions="Attempting to call tool",
        tool_calls=[
            ToolCall(
                name="python://failing_tool",
                call_id="call1",
                literal_args=json.dumps({"param": "value"}),
            )
        ],
        output=None,
    )

    # Step 2: return result after tool failure
    step2 = StepOutput(
        short_explanation="Handling error",
        instructions="Tool failed",
        tool_calls=[],
        output=MockResponse(answer="Error handled"),
    )

    llm_client.chat_completions.side_effect = [
        (step1, {"stats": "dummy"}),
        (step2, {"stats": "dummy"}),
    ]

    # Run agent
    result = await planning_agent.run(
        task_name="planning_task", task=task, max_iterations=5
    )

    # Verify result
    assert result.answer == "Error handled"

    # Verify the tool call was logged with errors
    async with session_maker() as db_session:
        from kavalai.agents.db import Task
        from sqlalchemy import select

        result = await db_session.execute(
            select(Task).where(Task.run_id == run.id).order_by(Task.created_at)
        )
        tasks = result.scalars().all()

        # Should have 4 tasks: failed tool call, step_0, step_1, overall agent task
        assert len(tasks) == 4

        # First task: failed tool call
        tool_task = tasks[0]
        assert tool_task.name == "python://failing_tool"
        assert tool_task.errors == ["ValidationError: Missing required field"]
        assert "Error:" in str(tool_task.output)

        # Second task: step 0
        step0_task = tasks[1]
        assert step0_task.name == "planning_task_step_0"

        # Third task: step 1
        step1_task = tasks[2]
        assert step1_task.name == "planning_task_step_1"

        # Fourth task: overall agent task
        agent_task = tasks[3]
        assert agent_task.name == "planning_task"
