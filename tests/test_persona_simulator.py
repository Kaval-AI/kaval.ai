import os
from unittest.mock import patch, MagicMock

import httpx
import pytest
import yaml

from kavalai.persona_simulator import run_simulation


@pytest.fixture
def temp_llm_profiles(tmp_path, monkeypatch):
    profile_dir = tmp_path / "llm_profiles"
    profile_dir.mkdir()

    config = {
        "name": "test_openai",
        "provider": "openai",
        "model_name": "gpt-4o-mini",
        "api_key": os.environ["OPENAI_API_KEY"],
    }

    profile_path = profile_dir / "test_openai.yaml"
    profile_path.write_text(yaml.dump(config))

    monkeypatch.setenv("LLM_PROFILES_PATH", str(profile_dir))
    return str(profile_dir)


@pytest.fixture
def mock_task_config(tmp_path):
    config = {
        "agent_server_url": "http://localhost:8000",
        "auth_username": "user",
        "auth_password": "password",
        "llm_profile_name": "test_openai",
        "task": "Say 'Hello, I am a test' and wait for the agent to respond.",
        "max_turns": 1,
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.dump(config))
    return str(path)


@pytest.fixture
def mock_persona_config(tmp_path):
    config = {
        "name": "Test Persona",
        "persona_description": "A test persona",
        "mood": "happy",
    }
    path = tmp_path / "persona.yaml"
    path.write_text(yaml.dump(config))
    return str(path)


@pytest.fixture
def mock_openapi_spec():
    return {
        "openapi": "3.0.0",
        "paths": {
            "/run_agent": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/InputType"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/OutputType"
                                    }
                                }
                            }
                        }
                    },
                }
            }
        },
        "components": {
            "schemas": {
                "InputType": {
                    "type": "object",
                    "properties": {"data": {"$ref": "#/components/schemas/input"}},
                },
                "OutputType": {
                    "type": "object",
                    "properties": {"data": {"$ref": "#/components/schemas/output"}},
                },
                "input": {
                    "type": "object",
                    "properties": {"user_message": {"type": "string"}},
                },
                "output": {
                    "type": "object",
                    "properties": {"agent_response": {"type": "string"}},
                },
            }
        },
    }


@pytest.mark.asyncio
async def test_run_simulation_real_llm(
    mock_task_config, mock_persona_config, mock_openapi_spec, temp_llm_profiles
):
    with patch("httpx.AsyncClient.get") as mock_get, patch(
        "httpx.AsyncClient.post"
    ) as mock_post, patch("kavalai.persona_simulator.chat_completions") as mock_chat:
        # Mock LLM response
        from kavalai.agents.db import ModelCallStat

        mock_chat.return_value = (
            MagicMock(
                thought="test thought",
                terminate=False,
                data=MagicMock(user_message="Hello, I am a test"),
            ),
            ModelCallStat(call_type="llm", model="test", duration_seconds=0.1),
        )

        # Mock OpenAPI spec fetch
        mock_get.return_value = MagicMock(spec=httpx.Response)
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_openapi_spec
        mock_get.return_value.raise_for_status = MagicMock()

        # Mock Agent response
        mock_post.return_value = MagicMock(spec=httpx.Response)
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "session_id": "session123",
            "data": {"agent_response": "Hello persona"},
        }
        mock_post.return_value.raise_for_status = MagicMock()

        await run_simulation(mock_task_config, mock_persona_config)

        assert mock_get.called
        assert mock_post.called
        assert mock_chat.called


@pytest.mark.asyncio
async def test_run_simulation_http_error(
    mock_task_config, mock_persona_config, mock_openapi_spec, temp_llm_profiles
):
    with patch("httpx.AsyncClient.get") as mock_get, patch(
        "httpx.AsyncClient.post"
    ) as mock_post, patch("kavalai.persona_simulator.chat_completions") as mock_chat:
        # Mock LLM response
        from kavalai.agents.db import ModelCallStat

        mock_chat.return_value = (
            MagicMock(
                thought="test thought",
                terminate=False,
                data=MagicMock(user_message="Hello, I am a test"),
            ),
            ModelCallStat(call_type="llm", model="test", duration_seconds=0.1),
        )

        mock_get.return_value = MagicMock(spec=httpx.Response)
        mock_get.return_value.json.return_value = mock_openapi_spec
        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status = MagicMock()

        # Simulate HTTP Error on agent call
        mock_post.return_value = MagicMock(spec=httpx.Response)
        mock_post.return_value.status_code = 500
        mock_post.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Error", request=MagicMock(), response=mock_post.return_value
        )

        await run_simulation(mock_task_config, mock_persona_config)

        assert mock_post.called
        assert mock_chat.called
