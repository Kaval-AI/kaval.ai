import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel

from kavalai.agents.client import AgentClient


class MockInput(BaseModel):
    user_message: str


class MockOutput(BaseModel):
    agent_response: str


@pytest.mark.asyncio
async def test_agent_client_discover_schemas():
    client = AgentClient("http://testserver")

    mock_openapi = {
        "paths": {
            "/run_agent": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "properties": {
                                        "data": {
                                            "properties": {
                                                "user_message": {"type": "string"}
                                            },
                                            "type": "object",
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "properties": {
                                            "data": {
                                                "properties": {
                                                    "agent_response": {"type": "string"}
                                                },
                                                "type": "object",
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                }
            }
        }
    }

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_openapi
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        await client.discover_schemas()

    assert client.input_schema is not None
    assert "user_message" in client.input_schema.model_fields
    assert client.output_schema is not None
    assert "agent_response" in client.output_schema.model_fields


@pytest.mark.asyncio
async def test_agent_client_run_agent():
    client = AgentClient("http://testserver")

    # Pre-set schemas to avoid discovery in this test
    client.input_schema = MockInput
    client.output_schema = MockOutput

    mock_response = {
        "session_id": "test-session",
        "data": {"agent_response": "Hello world"},
    }

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        data = MockInput(user_message="Hi")
        result = await client.run_agent(data)

        assert result.agent_response == "Hello world"
        assert client.session_id == "test-session"


@pytest.mark.asyncio
async def test_agent_client_auth_only_username():
    client = AgentClient("http://testserver", username="user")
    assert client.auth is None


@pytest.mark.asyncio
async def test_agent_client_auth_only_password():
    client = AgentClient("http://testserver", password="pass")
    assert client.auth is None
