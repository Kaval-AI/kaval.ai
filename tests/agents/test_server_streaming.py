import pytest
from fastapi.testclient import TestClient
from kavalai.agents.server import create_agent_app
from kavalai.agents.workflow import Workflow
import json


@pytest.fixture
def workflow():
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
    return Workflow.from_yaml(workflow_yaml)


@pytest.fixture
def app(workflow, agents_session_maker):
    return create_agent_app(workflow, session_provider=agents_session_maker)


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.mark.asyncio
async def test_stream_agent_endpoint(monkeypatch, app):
    # 1. Mock chat_completions to simulate streaming
    async def mock_chat_completions(
        model, response_model, messages, streamer=None, **kwargs
    ):
        if streamer:
            streamer.stream_partial('{"agent_response": "He')
            streamer.stream_partial('{"agent_response": "Hello world"}')
            streamer.stream_complete('{"agent_response": "Hello world"}')

        response = response_model(agent_response="Hello world")
        from kavalai.agents.db import ModelCallStat

        stats = ModelCallStat(
            call_type="llm", model=model, response_code=200, duration_seconds=0.1
        )
        return response, stats

    monkeypatch.setattr(
        "kavalai.agents.workflow.chat_completions", mock_chat_completions
    )

    # Use httpx.AsyncClient for testing streaming response
    from httpx import AsyncClient, ASGITransport

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as ac:
        input_data = {"data": {"user_message": "Junie"}}
        response = await ac.post("/stream_agent", json=input_data)

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/x-ndjson"

        lines = [line for line in response.text.split("\n") if line]
        assert len(lines) == 3

        partial1 = json.loads(lines[0])
        assert partial1["type"] == "partial"
        assert partial1["value"] == '{"agent_response": "He'

        partial2 = json.loads(lines[1])
        assert partial2["type"] == "partial"
        assert partial2["value"] == '{"agent_response": "Hello world"}'

        complete = json.loads(lines[2])
        assert complete["type"] == "complete"
