import json
import os
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import async_sessionmaker

from kavalai.agents.server import (
    create_agent_app,
    create_app_from_env_conf,
    run_agent_server,
)
from kavalai.agents.workflow import Workflow
from kavalai.agents.workflow_model import WorkflowModel


@pytest.fixture
def mock_workflow():
    model = WorkflowModel(
        name="test_agent",
        description="Test description",
        version="1.0.0",
        data_types={
            "input": {"type": "object", "properties": {"query": {"type": "string"}}},
            "output": {"type": "object", "properties": {"result": {"type": "string"}}},
        },
        tasks=[],
    )
    return Workflow(model)


@pytest.fixture
def mock_workflow_path(tmp_path):
    workflow_file = tmp_path / "workflow.yaml"
    content = """
name: test_agent
description: Test agent description
version: 1.0.0
data_types:
  input:
    type: object
    properties:
      query: {type: string}
  output:
    type: object
    properties:
      result: {type: string}
tasks: []
"""
    workflow_file.write_text(content)
    return str(workflow_file)


@pytest.fixture
def streaming_workflow():
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
    type: llm
    prompt: "Hello"
    inputs:
      user_message:
        type: context
        value: input.user_message
    output: output
    stream: true
"""
    return Workflow.from_yaml(workflow_yaml)


# --- Health & Liveness Tests (from test_server_health.py) ---


def test_liveness_endpoint(mock_workflow):
    app = create_agent_app(mock_workflow)
    client = TestClient(app)
    response = client.get("/liveness")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_health_endpoint_success(mock_workflow):
    mock_session = AsyncMock()
    mock_session.execute.return_value = None

    class MockAsyncContextManager:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, exc_type, exc, tb):
            pass

    mock_session_provider = MagicMock(spec=async_sessionmaker)
    mock_session_provider.return_value = MockAsyncContextManager()

    app = create_agent_app(mock_workflow, session_provider=mock_session_provider)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "connected"}
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_health_endpoint_failure(mock_workflow):
    class MockAsyncContextManager:
        async def __aenter__(self):
            raise Exception("DB Error")

        async def __aexit__(self, exc_type, exc, tb):
            pass

    mock_session_provider = MagicMock(spec=async_sessionmaker)
    mock_session_provider.return_value = MockAsyncContextManager()

    app = create_agent_app(mock_workflow, session_provider=mock_session_provider)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["detail"] == "Database connection failed"


# --- Config Tests (from test_server_config.py) ---


def test_create_app_from_env_conf_overrides(mock_workflow_path):
    env_vars = {
        "KAVALAI_AGENT_WORKFLOW_PATH": mock_workflow_path,
        "KAVALAI_DB_URI": "postgresql+asyncpg://user:pass@localhost/db",
        "KAVALAI_DB_SCHEMA": "public",
        "KAVALAI_DB_POOL_SIZE": "5",
        "KAVALAI_DB_MAX_OVERFLOW": "10",
        "KAVALAI_SQL_ECHO": "True",
        "KAVALAI_OPENAI_SERVICE_TIER": "priority",
    }

    with patch.dict(os.environ, env_vars), patch(
        "kavalai.agents.server.db_manager.get_sessionmaker"
    ) as mock_get_sessionmaker:
        app = create_app_from_env_conf()
        assert app.title == "test_agent"
        mock_get_sessionmaker.assert_called_with(
            uri="postgresql+asyncpg://user:pass@localhost/db",
            echo=True,
            pool_size=5,
            max_overflow=10,
        )

        mock_get_sessionmaker.reset_mock()
        app = create_app_from_env_conf(
            db_uri="sqlite+aiosqlite:///:memory:", sql_echo=False, pool_size=0
        )
        mock_get_sessionmaker.assert_called_with(
            uri="sqlite+aiosqlite:///:memory:",
            echo=False,
            pool_size=0,
            max_overflow=10,
        )


def test_run_agent_server_workflow_access(mock_workflow_path):
    env_vars = {
        "KAVALAI_AGENT_WORKFLOW_PATH": mock_workflow_path,
        "KAVALAI_DB_URI": "sqlite+aiosqlite:///:memory:",
    }

    with patch.dict(os.environ, env_vars), patch(
        "kavalai.agents.server.db_manager.get_sessionmaker"
    ), patch("kavalai.agents.server.uvicorn.run"):
        run_agent_server()


# --- Streaming Tests (from test_server_streaming.py) ---


@pytest.mark.asyncio
async def test_stream_agent_endpoint(
    monkeypatch, streaming_workflow, agents_session_maker
):
    app = create_agent_app(streaming_workflow, session_provider=agents_session_maker)

    async def mock_chat_completions(
        model, response_model, messages, streamer=None, **kwargs
    ):
        if streamer:
            await streamer.stream_partial('{"agent_response": "He')
            await streamer.stream_partial('{"agent_response": "Hello world"}')
            await streamer.stream_complete('{"agent_response": "Hello world"}')

        response = response_model(agent_response="Hello world")
        from kavalai.agents.db import ModelCallStat

        stats = ModelCallStat(
            call_type="llm",
            model="test-model",
            response_code=200,
            duration_seconds=0.1,
            cost=0,
        )
        return response, stats

    monkeypatch.setattr(
        "kavalai.agents.workflow.LLMClient.chat_completions", mock_chat_completions
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as ac:
        input_data = {"data": {"user_message": "Junie"}}
        response = await ac.post("/stream_agent", json=input_data)

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        lines = [
            line[6:] for line in response.text.split("\n") if line.startswith("data: ")
        ]
        assert len(lines) == 4

        partial1 = json.loads(lines[0])
        assert partial1["type"] == "partial"
        assert partial1["value"] == '{"agent_response": "He'

        partial2 = json.loads(lines[1])
        assert partial2["type"] == "partial"
        assert partial2["value"] == '{"agent_response": "Hello world"}'

        complete = json.loads(lines[2])
        assert complete["type"] == "complete"
        assert complete["name"] == "output"

        # Final output
        final_output = json.loads(lines[3])
        assert final_output["type"] == "complete"
        assert final_output["name"] == "output"
        final_value = json.loads(final_output["value"])
        assert "session_id" in final_value
        assert final_value["data"]["agent_response"] == "Hello world"


# --- Logging Tests (from test_server_logging.py) ---


@patch("kavalai.agents.server.env")
@patch("kavalai.agents.server.Workflow")
@patch("kavalai.agents.server.db_manager")
@patch("kavalai.agents.server.uvicorn.run")
@patch("kavalai.agents.server.logger")
def test_run_agent_server_logging(
    mock_logger, mock_uvicorn, mock_db_manager, mock_workflow_class, mock_env
):
    mock_env.side_effect = lambda key, default=None: {
        "KAVALAI_AGENT_WORKFLOW_PATH": "test_path.yaml",
        "KAVALAI_DB_URI": "postgresql://user:password@localhost/dbname",
        "KAVALAI_DB_SCHEMA": "test_schema",
        "KAVALAI_AGENT_BASIC_AUTH_USER": "admin",
        "KAVALAI_AGENT_BASIC_AUTH_USER_PASSWORD": "secret_password",
    }.get(key, default)

    mock_env.str.side_effect = lambda key, default=None: {
        "KAVALAI_AGENT_WORKFLOW_PATH": "test_path.yaml",
        "KAVALAI_AGENT_HOST": "0.0.0.0",
        "KAVALAI_AGENT_BASIC_AUTH_USER": "admin",
        "KAVALAI_AGENT_BASIC_AUTH_USER_PASSWORD": "secret_password",
    }.get(key, default)

    mock_env.int.side_effect = lambda key, default=None: {
        "KAVALAI_AGENT_PORT": 10000,
        "KAVALAI_DB_POOL_SIZE": 5,
        "KAVALAI_DB_MAX_OVERFLOW": 10,
    }.get(key, default)

    mock_env.bool.return_value = False

    mock_workflow_instance = MagicMock()
    mock_workflow_instance.workflow_model.name = "test_agent"
    mock_workflow_instance.get_data_type.return_value = str
    mock_workflow_class.from_yaml_path.return_value = mock_workflow_instance

    run_agent_server()

    info_logs = [call.args[0] for call in mock_logger.info.call_args_list]

    assert any("Database URI:" in log for log in info_logs)
    assert any("Database Schema: test_schema" in log for log in info_logs)
    assert any("Database Pool Size: 5" in log for log in info_logs)
    assert any("Database Max Overflow: 10" in log for log in info_logs)
    assert any("Basic Auth configured for user: admin" in log for log in info_logs)
    assert any("Basic Auth password:" in log for log in info_logs)

    db_log = [log for log in info_logs if "Database URI:" in log][0]
    assert "postgresql://user:***@localhost/dbname" in db_log
    assert "password" not in db_log

    auth_log = [log for log in info_logs if "Basic Auth password:" in log][0]
    assert "Basic Auth password: ***" in auth_log
    assert "secret_password" not in auth_log


@patch("kavalai.agents.server.env")
@patch("kavalai.agents.server.Workflow")
@patch("kavalai.agents.server.db_manager")
@patch("kavalai.agents.server.uvicorn.run")
@patch("kavalai.agents.server.logger")
def test_run_agent_server_no_auth_warning(
    mock_logger, mock_uvicorn, mock_db_manager, mock_workflow_class, mock_env
):
    mock_env.side_effect = lambda key, default=None: {
        "KAVALAI_AGENT_WORKFLOW_PATH": "test_path.yaml",
        "KAVALAI_DB_URI": "postgresql://user:password@localhost/dbname",
        "KAVALAI_DB_SCHEMA": "test_schema",
        "KAVALAI_AGENT_BASIC_AUTH_USER": "",
        "KAVALAI_AGENT_BASIC_AUTH_USER_PASSWORD": "",
    }.get(key, default)

    mock_env.str.side_effect = lambda key, default=None: {
        "KAVALAI_AGENT_WORKFLOW_PATH": "test_path.yaml",
        "KAVALAI_AGENT_HOST": "0.0.0.0",
        "KAVALAI_AGENT_BASIC_AUTH_USER": "",
        "KAVALAI_AGENT_BASIC_AUTH_USER_PASSWORD": "",
    }.get(key, default)

    mock_env.int.side_effect = lambda key, default=None: {
        "KAVALAI_AGENT_PORT": 10000,
    }.get(key, default)

    mock_env.bool.return_value = False

    mock_workflow_instance = MagicMock()
    mock_workflow_instance.workflow_model.name = "test_agent"
    mock_workflow_instance.get_data_type.return_value = str
    mock_workflow_class.from_yaml_path.return_value = mock_workflow_instance

    run_agent_server()

    warning_logs = [call.args[0] for call in mock_logger.warning.call_args_list]
    assert any("Basic Auth is NOT configured" in log for log in warning_logs)
