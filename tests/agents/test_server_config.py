import os
from unittest.mock import patch

import pytest

from kavalai.agents.server import create_app_from_env_conf


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


def test_create_app_from_env_conf_overrides(mock_workflow_path):
    # Setup environment variables
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
        # Test 1: Use environment variables
        app = create_app_from_env_conf()
        assert app.title == "test_agent"
        mock_get_sessionmaker.assert_called_with(
            uri="postgresql+asyncpg://user:pass@localhost/db",
            echo=True,
            pool_size=5,
            max_overflow=10,
        )

        # Test 2: Override some values
        mock_get_sessionmaker.reset_mock()
        app = create_app_from_env_conf(
            db_uri="sqlite+aiosqlite:///:memory:", sql_echo=False, pool_size=0
        )
        mock_get_sessionmaker.assert_called_with(
            uri="sqlite+aiosqlite:///:memory:",
            echo=False,
            pool_size=0,
            max_overflow=10,  # From env
        )


def test_run_agent_server_workflow_access(mock_workflow_path):
    # Setup environment variables
    env_vars = {
        "KAVALAI_AGENT_WORKFLOW_PATH": mock_workflow_path,
        "KAVALAI_DB_URI": "sqlite+aiosqlite:///:memory:",
    }

    with patch.dict(os.environ, env_vars), patch(
        "kavalai.agents.server.db_manager.get_sessionmaker"
    ), patch("kavalai.agents.server.uvicorn.run"):
        from kavalai.agents.server import run_agent_server

        # This should not raise Unresolved reference or AttributeError
        # We need to capture logs to verify the message, but for now just check it runs
        run_agent_server()
