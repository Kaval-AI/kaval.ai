import pytest
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel

from kavalai.agents.agent import Agent, ToolCall, get_step_output_type
from kavalai.agents.run_context import RunContext
from kavalai.functionkernel import FunctionKernel
from kavalai.llm_clients.base_client import BaseLlmClient


class MockResponse(BaseModel):
    answer: str


@pytest.fixture
def mock_kernel():
    kernel = MagicMock(spec=FunctionKernel)
    kernel.get_tool_descriptions = AsyncMock(return_value="Mock tool descriptions")
    kernel.call_tool = AsyncMock(return_value="Tool result")
    return kernel


@pytest.fixture
def mock_llm_client():
    client = MagicMock(spec=BaseLlmClient)
    client.chat_completions = AsyncMock()
    return client


@pytest.fixture
def run_context():
    return RunContext(data={})


@pytest.mark.asyncio
async def test_prompt_tool_call_then_output(mock_kernel, mock_llm_client, run_context):
    """A tool-calling step followed by a step producing the final output."""
    agent = Agent(
        llm_client=mock_llm_client,
        kernel=mock_kernel,
        run_context=run_context,
    )

    StepOutput = get_step_output_type(MockResponse)
    step1 = StepOutput(
        instructions="call the test tool",
        tool_calls=[
            ToolCall(name="python://test_tool", literal_args='{"val": 1}', call_id="c1")
        ],
    )
    step2 = StepOutput(
        instructions="return the final answer",
        output=MockResponse(answer="Final answer"),
    )
    mock_llm_client.chat_completions.side_effect = [step1, step2]

    result = await agent.prompt(
        "Do something", response_model=MockResponse, max_steps=5
    )

    assert isinstance(result, MockResponse)
    assert result.answer == "Final answer"
    # The tool was executed with the resolved literal arguments.
    mock_kernel.call_tool.assert_awaited_once_with(
        tool_uri="python://test_tool", arguments={"val": 1}
    )
    # Only two LLM calls were needed (stopped once output produced).
    assert mock_llm_client.chat_completions.await_count == 2


@pytest.mark.asyncio
async def test_prompt_plain_string_output(mock_kernel, mock_llm_client, run_context):
    """Without a response_model the agent returns a plain string output."""
    agent = Agent(
        llm_client=mock_llm_client,
        kernel=mock_kernel,
        run_context=run_context,
    )

    StepOutput = get_step_output_type(str)
    mock_llm_client.chat_completions.side_effect = [
        StepOutput(instructions="greet the user", output="hello there")
    ]

    result = await agent.prompt("Greet the user")

    assert result == "hello there"
    mock_kernel.call_tool.assert_not_called()


@pytest.mark.asyncio
async def test_prompt_respects_max_steps(mock_kernel, mock_llm_client, run_context):
    """The loop stops after max_steps even if no final output is produced."""
    agent = Agent(
        llm_client=mock_llm_client,
        kernel=mock_kernel,
        run_context=run_context,
    )

    StepOutput = get_step_output_type(MockResponse)
    # Always request a tool call, never produce an output.
    looping_step = StepOutput(
        instructions="keep looping",
        tool_calls=[ToolCall(name="python://loop", call_id="c1")],
    )
    mock_llm_client.chat_completions.side_effect = [looping_step] * 10

    result = await agent.prompt(
        "Never finish", response_model=MockResponse, max_steps=3
    )

    assert result is None
    assert mock_llm_client.chat_completions.await_count == 3
    assert mock_kernel.call_tool.await_count == 3


@pytest.mark.asyncio
async def test_resolve_args_merges_sources(mock_kernel, mock_llm_client):
    """literal/context/input args merge with literal > context > input precedence."""
    agent = Agent(
        llm_client=mock_llm_client,
        kernel=mock_kernel,
        run_context=RunContext(data={"user_id": "u-123"}),
    )

    tool_call = ToolCall(
        name="python://test",
        literal_args='{"mode": "fast"}',
        planner_context_args='{"prev": "c1"}',
        input_args='{"uid": "user_id"}',
    )
    planner_context = {"c1": "previous result"}

    args = agent._resolve_args(tool_call, planner_context)

    assert args == {
        "uid": "u-123",
        "prev": "previous result",
        "mode": "fast",
    }


@pytest.mark.asyncio
async def test_planner_context_args_resolve_from_planner_context(
    mock_kernel, mock_llm_client
):
    """planner_context_args resolves against the per-invocation planner_context."""
    agent = Agent(
        llm_client=mock_llm_client,
        kernel=mock_kernel,
        run_context=RunContext(data={"input_key": "input value"}),
    )

    tool_call = ToolCall(
        name="python://test",
        planner_context_args='{"a": "c1", "b": "missing"}',
    )
    planner_context = {"c1": "tool result"}

    args = agent._resolve_args(tool_call, planner_context)

    # "c1" comes from planner_context; an unknown key resolves to None and
    # input data is not reachable through planner_context_args.
    assert args == {"a": "tool result", "b": None}


@pytest.mark.asyncio
async def test_planner_context_isolated_across_invocations(
    mock_kernel, mock_llm_client, run_context
):
    """Tool results in planner_context do not leak into the next invocation."""
    agent = Agent(
        llm_client=mock_llm_client,
        kernel=mock_kernel,
        run_context=run_context,
    )

    StepOutput = get_step_output_type(MockResponse)

    # First invocation: produce a tool result under call_id "c1", then finish.
    mock_llm_client.chat_completions.side_effect = [
        StepOutput(
            instructions="run the tool",
            tool_calls=[ToolCall(name="python://t", call_id="c1")],
        ),
        StepOutput(instructions="finish", output=MockResponse(answer="first")),
    ]
    await agent.prompt("first", response_model=MockResponse)

    # Second invocation references "c1" which belonged to the previous run.
    second_call = ToolCall(
        name="python://t",
        planner_context_args='{"x": "c1"}',
        call_id="c2",
    )
    mock_llm_client.chat_completions.side_effect = [
        StepOutput(instructions="run the tool", tool_calls=[second_call]),
        StepOutput(instructions="finish", output=MockResponse(answer="second")),
    ]
    mock_kernel.call_tool.reset_mock()
    await agent.prompt("second", response_model=MockResponse)

    # The reference to the prior invocation's call_id resolves to None.
    first_args = mock_kernel.call_tool.await_args_list[0].kwargs["arguments"]
    assert first_args == {"x": None}


@pytest.mark.asyncio
async def test_tool_error_is_captured(mock_kernel, mock_llm_client, run_context):
    """A failing tool call is captured as a result string instead of raising."""
    agent = Agent(
        llm_client=mock_llm_client,
        kernel=mock_kernel,
        run_context=run_context,
    )
    mock_kernel.call_tool.side_effect = RuntimeError("boom")

    tool_call = ToolCall(name="python://broken", call_id="c1")
    _, _, result = await agent._call_tool(tool_call, {})

    assert result == "Error: boom"
