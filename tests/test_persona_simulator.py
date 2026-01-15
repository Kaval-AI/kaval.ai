from unittest.mock import patch, MagicMock, AsyncMock

import httpx
import pytest
import yaml

from kavalai.persona_simulator import run_simulation


@pytest.fixture
def mock_task_config(tmp_path):
    config = {
        "agent_server_url": "http://localhost:8000",
        "auth_username": "user",
        "auth_password": "password",
        "llm_provider": "openai",
        "task": "Test Task",
        "max_turns": 2,
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
async def test_run_simulation_success(
    mock_task_config, mock_persona_config, mock_openapi_spec
):
    with patch("httpx.AsyncClient.get") as mock_get, patch(
        "httpx.AsyncClient.post"
    ) as mock_post, patch("instructor.from_provider") as mock_instructor:
        # Mock OpenAPI spec fetch
        mock_get.return_value = MagicMock(spec=httpx.Response)
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_openapi_spec
        mock_get.return_value.raise_for_status = MagicMock()

        # Mock LLM client
        mock_llm = AsyncMock()
        mock_instructor.return_value = mock_llm

        # Mock LLM response
        mock_persona_resp = MagicMock()
        mock_persona_resp.thought = "I should say hello"
        mock_persona_resp.terminate = False
        mock_persona_resp.data = MagicMock()
        mock_persona_resp.data.user_message = "Hello agent"
        mock_persona_resp.data.model_dump.return_value = {"user_message": "Hello agent"}

        mock_llm.chat.completions.create.return_value = mock_persona_resp

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
        assert mock_llm.chat.completions.create.called


@pytest.mark.asyncio
async def test_run_simulation_terminate(
    mock_task_config, mock_persona_config, mock_openapi_spec
):
    with patch("httpx.AsyncClient.get") as mock_get, patch(
        "httpx.AsyncClient.post"
    ) as mock_post, patch("instructor.from_provider") as mock_instructor:
        mock_get.return_value = MagicMock(spec=httpx.Response)
        mock_get.return_value.json.return_value = mock_openapi_spec

        mock_llm = AsyncMock()
        mock_instructor.return_value = mock_llm

        mock_persona_resp = MagicMock()
        mock_persona_resp.terminate = True
        mock_llm.chat.completions.create.return_value = mock_persona_resp

        await run_simulation(mock_task_config, mock_persona_config)

        assert mock_post.call_count == 0


@pytest.mark.asyncio
async def test_run_simulation_http_error(
    mock_task_config, mock_persona_config, mock_openapi_spec
):
    with patch("httpx.AsyncClient.get") as mock_get, patch(
        "httpx.AsyncClient.post"
    ) as mock_post, patch("instructor.from_provider") as mock_instructor:
        mock_get.return_value = MagicMock(spec=httpx.Response)
        mock_get.return_value.json.return_value = mock_openapi_spec

        mock_llm = AsyncMock()
        mock_instructor.return_value = mock_llm

        mock_persona_resp = MagicMock()
        mock_persona_resp.terminate = False
        mock_persona_resp.data = MagicMock()
        mock_persona_resp.data.user_message = "Hello"
        mock_persona_resp.data.model_dump.return_value = {"user_message": "Hello"}
        mock_llm.chat.completions.create.return_value = mock_persona_resp

        # Simulate HTTP Error
        mock_post.return_value = MagicMock(spec=httpx.Response)
        mock_post.return_value.status_code = 500
        mock_post.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Error", request=MagicMock(), response=mock_post.return_value
        )

        await run_simulation(mock_task_config, mock_persona_config)

        assert mock_post.called
