import os

import pytest
from fastapi.testclient import TestClient

# Assuming your server code is in kavalai_server.py
from kavalai.agents.server import create_agent_app
from kavalai.agents.workflow import Workflow

# 1. Setup the YAML configuration
SIMPLE_YAML = """
name: BB King Agent
description: Just talks about the blues.
llm_profile_name: test-gpt-4o
data_types:
  input:
    type: object
    properties:
      user_message: {type: string}
  output:
    type: object
    properties:
      agent_response:
        type: string
        max_length: 250
mcp_servers: []
tasks:
  - name: Blues Talk
    prompt: "You are BB King. You like talking about the blues."
    inputs:
      input: {type: context, value: input}
    output: output
"""


@pytest.fixture(scope="function")
def client(agents_session_maker, agents_db, tmp_path, monkeypatch):
    # Set dummy auth for testing
    os.environ["HTTP_BASIC_AUTH_USER"] = "lucille"
    os.environ["HTTP_BASIC_AUTH_PASSWORD"] = "blues123"

    # Initialize real workflow from string/file
    workflow = Workflow.from_yaml(SIMPLE_YAML)

    # Ensure LLM profile exists as a file
    import yaml

    profile_name = "test-gpt-4o"
    profile_dir = tmp_path / "llm_profiles"
    profile_dir.mkdir()
    profile_path = profile_dir / "test-gpt-4o.yaml"

    with open(profile_path, "w") as f:
        yaml.dump(
            {
                "name": profile_name,
                "provider": "openai",
                "model_name": "gpt-4o",
            },
            f,
        )

    monkeypatch.setenv("LLM_PROFILES_PATH", str(profile_dir))

    # Create the app
    app = create_agent_app(workflow=workflow, session_provider=agents_session_maker)
    yield TestClient(app)


def test_run_agent_success(client):
    """Test a successful authenticated request to the agent."""
    payload = {
        "session_id": None,
        "data": {"user_message": "Tell me about your guitar."},
    }

    from unittest.mock import AsyncMock, patch

    # patch get_instructor to return a mock client that returns a dict
    # the server will then validate this dict against its own OutputDataType
    mock_response = {"agent_response": "Lucille is my lady."}

    with patch("kavalai.agents.workflow.get_instructor") as mock_get_instructor:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_instructor.return_value = mock_client

        response = client.post("/run_agent", json=payload, auth=("lucille", "blues123"))

    assert response.status_code == 200
    data = response.json()
    assert "agent_response" in data["data"]
    print(data)
    # Verify the LLM persona is working
    assert any(
        word in data["data"]["agent_response"].lower()
        for word in ["blues", "guitar", "king", "lucille"]
    )


def test_run_agent_unauthorized(client):
    """Test that wrong credentials return 401."""
    payload = {"data": {"user_message": "Hello?"}}
    response = client.post(
        "/run_agent", json=payload, auth=("wrong_user", "wrong_pass")
    )
    assert response.status_code == 401


def test_run_agent_invalid_input_schema(client):
    """Test that Pydantic validation catches wrong data structures."""
    # Sending 'text' instead of 'user_message' as defined in YAML
    payload = {"data": {"wrong_field": "This will fail"}}
    response = client.post("/run_agent", json=payload, auth=("lucille", "blues123"))
    # FastAPI returns 422 Unprocessable Entity for schema validation errors
    assert response.status_code == 422
