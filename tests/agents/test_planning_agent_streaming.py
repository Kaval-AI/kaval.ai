import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel
from kavalai.agents.planning_agent import PlanningAgent, get_step_output_type
from kavalai.functionkernel import FunctionKernel
from kavalai.agents.run_context import RunContext
from kavalai.llm_clients.llm_client import LLMClient
from kavalai.llm_clients.common import Streamer


class MockResponse(BaseModel):
    answer: str


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
