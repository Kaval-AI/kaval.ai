import json
import pytest

from kavalai.agents.workflow import Workflow
from kavalai.agents.workflow_model import (
    WorkflowModel,
    McpTask,
    ArgumentInfo,
    McpServer,
)


class FakeResponse:
    def __init__(self, content):
        self.content = content
        self.isError = False


class FakeTextContent:
    def __init__(self, text: str):
        self.text = text


class FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def initialize(self):
        return None

    async def call_tool(self, tool_name: str, arguments: dict):
        # Echo back the arguments as JSON string
        return FakeResponse(
            [FakeTextContent(json.dumps({"result": arguments.get("message", "")}))]
        )


class FakeStdioClient:
    def __init__(self, *_args, **_kwargs):
        pass

    async def __aenter__(self):
        # Return (read, write) placeholders
        return (None, None)

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_workflow_mcp_tool_call(monkeypatch):
    # Monkeypatch mcp stdio_client and ClientSession to fakes
    import kavalai.functionkernel as fk

    def fake_stdio_client(_params):
        return FakeStdioClient()

    def fake_client_session(_read, _write):
        return FakeSession()

    monkeypatch.setattr(fk, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(fk, "ClientSession", fake_client_session)

    workflow_model = WorkflowModel(
        name="mcp-demo",
        description="Test MCP tool call",
        data_types={
            "input": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
            },
            "output": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
            },
        },
        rest_servers=[],
        mcp_servers=[McpServer(name="demo", command="dummy")],
        tasks=[
            McpTask(
                name="call_mcp",
                tool="echo",
                mcp_server="demo",
                inputs={
                    "message": ArgumentInfo(type="context", value="input.message"),
                },
                output="output",
            )
        ],
    )

    workflow = Workflow(workflow_model)

    result = await workflow.run({"message": "hello"})

    assert result.data is not None
    assert result.data.result == "hello"


@pytest.mark.asyncio
async def test_workflow_mcp_tool_call_env(monkeypatch):
    monkeypatch.setenv("MCP_CMD", "dummy-env")

    # Monkeypatch mcp stdio_client and ClientSession to fakes
    import kavalai.functionkernel as fk

    last_command = None

    def fake_stdio_client(params):
        nonlocal last_command
        last_command = params.command
        return FakeStdioClient()

    def fake_client_session(_read, _write):
        return FakeSession()

    monkeypatch.setattr(fk, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(fk, "ClientSession", fake_client_session)

    workflow_model = WorkflowModel(
        name="mcp-demo-env",
        description="Test MCP tool call with env",
        data_types={
            "input": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
            },
            "output": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
            },
        },
        rest_servers=[],
        mcp_servers=[McpServer(name="demo", command_env="MCP_CMD")],
        tasks=[
            McpTask(
                name="call_mcp",
                tool="echo",
                mcp_server="demo",
                inputs={
                    "message": ArgumentInfo(type="context", value="input.message"),
                },
                output="output",
            )
        ],
    )

    workflow = Workflow(workflow_model)

    result = await workflow.run({"message": "hello"})

    assert result.data is not None
    assert result.data.result == "hello"
    assert last_command == "dummy-env"
