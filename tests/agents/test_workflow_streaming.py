import pytest
import asyncio
from pydantic import BaseModel
from kavalai.agents.workflow import Workflow
from kavalai.llm_clients.common import StreamContent


class InputModel(BaseModel):
    user_message: str


class OutputModel(BaseModel):
    agent_response: str


@pytest.mark.asyncio
async def test_workflow_streaming_prompt(monkeypatch):
    # 1. Setup Workflow
    workflow_yaml = """
name: StreamTest
description: Test streaming
data_types:
  input:
    type: object
    properties:
      user_message:
        type: string
  output:
    type: object
    properties:
      agent_response:
        type: string
tasks:
  - name: generate
    prompt: "Hello {{input.user_message}}"
    output: output
    stream: true
"""
    workflow = Workflow.from_yaml(workflow_yaml)

    # 2. Mock chat_completions to simulate streaming
    async def mock_chat_completions(
        model, response_model, messages, streamer=None, **kwargs
    ):
        if streamer:
            # Simulate partial stream
            await streamer.stream_partial('{"agent_response": "He')
            await streamer.stream_partial('{"agent_response": "Hello world"}')
            await streamer.stream_complete('{"agent_response": "Hello world"}')

        response = response_model(agent_response="Hello world")
        stats = None  # Not needed for this test
        return response, stats

    monkeypatch.setattr(
        "kavalai.agents.workflow.chat_completions", mock_chat_completions
    )

    # 3. Run Workflow with stream
    queue = asyncio.Queue()
    input_data = {"user_message": "Junie"}
    task = asyncio.create_task(workflow.run(input_data=input_data, queue=queue))

    # 4. Verify Stream Content
    lines = []
    while not task.done() or not queue.empty():
        try:
            line = await asyncio.wait_for(queue.get(), timeout=0.1)
            lines.append(line)
        except asyncio.TimeoutError:
            continue

    result = await task

    assert len(lines) == 3

    partial1 = StreamContent.model_validate_json(lines[0])
    assert partial1.type == "partial"
    assert partial1.name == "output"
    assert partial1.value == '{"agent_response": "He'

    partial2 = StreamContent.model_validate_json(lines[1])
    assert partial2.type == "partial"
    assert partial2.name == "output"
    assert partial2.value == '{"agent_response": "Hello world"}'

    complete = StreamContent.model_validate_json(lines[2])
    assert complete.type == "complete"
    assert complete.name == "output"
    assert complete.value == ""  # Streamer.stream_complete sets value to ""

    assert result.data.agent_response == "Hello world"


@pytest.mark.asyncio
async def test_workflow_streaming_tool(monkeypatch):
    # 1. Setup Workflow with tool
    workflow_yaml = """
name: ToolStreamTest
description: Test tool streaming
rest_servers:
  - name: mock
    url: "http://mock"
data_types:
  input:
    type: object
    properties:
      user_message:
        type: string
  output:
    type: object
    properties:
      agent_response:
        type: string
tasks:
  - name: tool_call
    tool: "test"
    rest_server: mock
    inputs:
      msg:
        type: context
        value: input.user_message
    output: output
    stream: true
"""
    workflow = Workflow.from_yaml(workflow_yaml)

    # 2. Mock httpx.AsyncClient.request
    class MockResponse:
        def __init__(self, json_data):
            self.json_data = json_data

        def json(self):
            return self.json_data

        def raise_for_status(self):
            pass

    async def mock_request(*args, **kwargs):
        return MockResponse({"agent_response": "Tool response"})

    monkeypatch.setattr("httpx.AsyncClient.request", mock_request)

    # 3. Run Workflow with stream
    queue = asyncio.Queue()
    input_data = {"user_message": "Junie"}
    task = asyncio.create_task(workflow.run(input_data=input_data, queue=queue))

    # 4. Verify Stream Content
    lines = []
    while not task.done() or not queue.empty():
        try:
            line = await asyncio.wait_for(queue.get(), timeout=0.1)
            lines.append(line)
        except asyncio.TimeoutError:
            continue

    result = await task

    # 4. Verify Stream Content
    assert len(lines) >= 1

    complete = StreamContent.model_validate_json(lines[-1])
    assert complete.type == "complete"
    assert complete.name == "output"
    assert complete.value == ""

    assert result.data.agent_response == "Tool response"
