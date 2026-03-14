import json
import pytest
import kavalai.functionkernel as fk
from kavalai.agents.workflow import Workflow
from kavalai.agents.workflow_model import (
    WorkflowModel,
    McpTask,
    TypeInputInfo,
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
    def __init__(self):
        self.initialize_called = 0
        self.call_count = 0
        self.aenter_called = 0
        self.aexit_called = 0

    async def __aenter__(self):
        self.aenter_called += 1
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.aexit_called += 1
        return False

    async def initialize(self):
        self.initialize_called += 1
        return None

    async def call_tool(self, tool_name: str, arguments: dict):
        self.call_count += 1
        return FakeResponse(
            [FakeTextContent(json.dumps({"result": arguments.get("message", "")}))]
        )


class FakeClient:
    def __init__(self, session):
        self.session = session
        self.aenter_called = 0
        self.aexit_called = 0

    async def __aenter__(self):
        self.aenter_called += 1
        return (None, None)

    async def __aexit__(self, exc_type, exc, tb):
        self.aexit_called += 1
        return False


@pytest.mark.asyncio
async def test_workflow_mcp_session_reuse(monkeypatch):
    session = FakeSession()
    client = FakeClient(session)

    def fake_stdio_client(params):
        return client

    def fake_client_session(_read, _write):
        return session

    monkeypatch.setattr(fk, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(fk, "ClientSession", fake_client_session)

    workflow_model = WorkflowModel(
        name="mcp-reuse",
        data_types={
            "input": {"type": "object", "properties": {"msg": {"type": "string"}}},
            "output1": {"type": "object", "properties": {"result": {"type": "string"}}},
            "output": {"type": "object", "properties": {"result": {"type": "string"}}},
        },
        mcp_servers=[McpServer(name="demo", command="dummy")],
        tasks=[
            McpTask(
                name="call1",
                tool="echo",
                mcp_server="demo",
                inputs={"message": TypeInputInfo(type="context", value="input.msg")},
                output="output1",
            ),
            McpTask(
                name="call2",
                tool="echo",
                mcp_server="demo",
                inputs={"message": TypeInputInfo(type="context", value="input.msg")},
                output="output",
            ),
        ],
    )

    workflow = Workflow(workflow_model)
    await workflow.run({"msg": "hello"})

    # Check that initialize was called only once
    assert session.initialize_called == 1
    # Check that call_tool was called twice
    assert session.call_count == 2
    # Check that client and session were entered once
    assert client.aenter_called == 1
    assert session.aenter_called == 1
    # Check that cleanup was called at the end
    assert client.aexit_called == 1
    assert session.aexit_called == 1


@pytest.mark.asyncio
async def test_workflow_mcp_http_support(monkeypatch):
    session = FakeSession()
    client = FakeClient(session)
    last_url = None

    def fake_sse_client(url):
        nonlocal last_url
        last_url = url
        return client

    def fake_client_session(_read, _write):
        return session

    monkeypatch.setattr(fk, "sse_client", fake_sse_client)
    monkeypatch.setattr(fk, "ClientSession", fake_client_session)

    workflow_model = WorkflowModel(
        name="mcp-http",
        data_types={
            "input": {"type": "object", "properties": {"msg": {"type": "string"}}},
            "output": {"type": "object", "properties": {"result": {"type": "string"}}},
        },
        mcp_servers=[McpServer(name="demo", url="http://example.com/sse")],
        tasks=[
            McpTask(
                name="call",
                tool="echo",
                mcp_server="demo",
                inputs={"message": TypeInputInfo(type="context", value="input.msg")},
                output="output",
            )
        ],
    )

    workflow = Workflow(workflow_model)
    await workflow.run({"msg": "hello"})

    assert last_url == "http://example.com/sse"
    assert session.call_count == 1
    assert client.aexit_called == 1
