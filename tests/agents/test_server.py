import pytest
from unittest.mock import MagicMock, patch
from fastmcp import Client
from kavalai.agents.workflow import Workflow
from kavalai.agents.server import create_mcp_agent_server

# Minimal YAML for testing prompt logic
SIMPLE_YAML = """
name: BB King Agent
description: Just talks about the blues.
llm_provider: openai/gpt-4o
data_types:
  input:
    type: object
    properties:
      user_message: {type: string}
  output:
    type: object
    properties:
      agent_response: {type: string}
mcp_servers: []
tasks:
  - name: Blues Talk
    prompt: "You are BB King. You like talking about the blues."
    inputs:
      input: {type: context, value: input}
    output: output
"""


@pytest.mark.asyncio
async def test_workflow_mcp_integration():
    workflow = Workflow.from_yaml(SIMPLE_YAML)
    mcp_server = create_mcp_agent_server(workflow)

    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "run_agent",
            {"input_data": {"data": {"user_message": "What is the meaning of life?"}}},
        )
        assert result.data.session_id
        assert result.data.data


def test_auth_logic_standalone():
    """
    Since we can't easily wrap the FastMCP Client in HTTP Basic Auth
    without the FastAPI layer, we test the auth function as a unit.
    """
    from fastapi import HTTPException
    from kavalai.agents.server import authenticate

    mock_creds = MagicMock()
    mock_creds.username = "admin"
    mock_creds.password = "password"

    with patch.dict(
        "os.environ",
        {"BASIC_AUTH_USERNAME": "admin", "BASIC_AUTH_PASSWORD": "password"},
    ):
        # Should succeed
        assert authenticate(mock_creds) == "admin"

        # Should fail
        mock_creds.password = "wrong"
        with pytest.raises(HTTPException) as exc:
            authenticate(mock_creds)
        assert exc.value.status_code == 401
