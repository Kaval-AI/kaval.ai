import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel

from kavalai.agents.v2.agent import Agent, ToolCall, get_step_output_type
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
    # Using chat_completions directly for non-streaming test
    client.chat_completions = AsyncMock()
    return client


@pytest.fixture
def run_context():
    return RunContext(data={})


@pytest.mark.asyncio
async def test_agent_run_basic(mock_kernel, mock_llm_client, run_context):
    input_data = {"user_id": "123"}
    agent = Agent(
        kernel=mock_kernel,
        run_context=run_context,
        llm_client=mock_llm_client,
        input_data=input_data,
        response_model=MockResponse,
    )

    StepOutput = get_step_output_type(MockResponse)

    # Mock LLM returning a step with a tool call, then a final output
    step1 = StepOutput(
        short_explanation="Calling tool",
        instructions="Use tool",
        tool_calls=[
            ToolCall(name="python://test_tool", literal_args='{"val": 1}', call_id="c1")
        ],
    )
    step2 = StepOutput(
        short_explanation="Done",
        instructions="Finish",
        output=MockResponse(answer="Final answer"),
    )

    mock_llm_client.chat_completions.side_effect = [step1, step2]

    result = await agent.run(task_name="test_task", task="Do something")

    assert result.answer == "Final answer"
    assert mock_kernel.call_tool.called
    assert "c1" in agent._planner_context
    assert agent._planner_context["c1"] == "Tool result"


@pytest.mark.asyncio
async def test_agent_templater():
    from kavalai.agents.v2.agent import AgentTemplater

    planner_context = {"c1": "result1"}
    input_data = {"i1": "input1"}
    templater = AgentTemplater(planner_context, input_data)

    assert templater.resolve("{{context.c1}}") == "result1"
    assert templater.resolve("{{input.i1}}") == "input1"
    assert templater.resolve({"key": "{{context.c1}}"}) == {"key": "result1"}
    assert templater.resolve(["{{input.i1}}"]) == ["input1"]
    assert templater.resolve("plain") == "plain"


@pytest.mark.asyncio
async def test_agent_run_streaming_bridge(mock_kernel, mock_llm_client, run_context):
    from kavalai.llm_clients.streamer import StreamContent as LlmStreamContent

    input_data = {}

    # Mock streamer from common.py
    from kavalai.llm_clients.common import Streamer as CommonStreamer

    mock_common_queue = asyncio.Queue()
    common_streamer = CommonStreamer(name="test", queue=mock_common_queue)

    agent = Agent(
        kernel=mock_kernel,
        run_context=run_context,
        llm_client=mock_llm_client,
        input_data=input_data,
        response_model=MockResponse,
        streamer=common_streamer,
        stream_output=True,
    )

    StepOutput = get_step_output_type(MockResponse)
    _ = StepOutput(
        short_explanation="Done",
        instructions="Finish",
        output=MockResponse(answer="Final answer"),
    )

    # Mock stream_chat_completions
    class AsyncIter:
        def __init__(self, items):
            self.items = items

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.items:
                raise StopAsyncIteration
            return self.items.pop(0)

    items = [
        LlmStreamContent(
            type="partial",
            name="response",
            value='{"short_explanation": "Done", "instructions": "Final answer", "output": {"answer": "Final',
        ),
        LlmStreamContent(type="partial", name="response", value=' answer"}}'),
        LlmStreamContent(
            type="complete",
            name="response",
            value='{"short_explanation": "Done", "instructions": "Final answer", "output": {"answer": "Final answer"}}',
        ),
    ]
    mock_llm_client.stream_chat_completions.return_value = AsyncIter(items)

    result = await agent.run(task_name="test_task", task="Do something")

    assert result.answer == "Final answer"

    # Check if common streamer received partials
    streamed_items = []
    while not mock_common_queue.empty():
        streamed_items.append(await mock_common_queue.get())

    assert any("Final" in item for item in streamed_items)
    assert any("answer" in item for item in streamed_items)
