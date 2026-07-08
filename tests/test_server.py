from fastapi.testclient import TestClient

from kavalai.server import create_agent_app
from kavalai.workflow import WorkflowEngine
from kavalai.llm_clients.base_client import BaseLlmClient

YAML = """
name: srv
description: test server
llm_model: openai/fake
data_types:
  input:
    type: object
    properties:
      user_message: {type: string}
  output:
    type: object
    properties:
      agent_response: {type: string}
nodes:
  - {name: start, type: start, next: reply}
  - name: reply
    type: llm
    prompt: hi
    inputs: {input: {type: context, value: input}}
    output: output
    next: end
  - {name: end, type: end, output: output}
"""


class FakeClient(BaseLlmClient):
    def __init__(self, *args, **kwargs):
        super().__init__()

    async def chat_completions(self, *, chat_history, response_model=None):
        return response_model(**{name: "hello" for name in response_model.model_fields})


def _factory(model, parameters=None, stats_receiver=None):
    return FakeClient()


def make_client_app() -> TestClient:
    engine = WorkflowEngine.from_yaml(YAML, client_factory=_factory)
    app = create_agent_app(engine=engine, auth_dependency=lambda: None)
    return TestClient(app)


def test_run_agent_returns_output():
    client = make_client_app()
    resp = client.post("/run_agent", json={"data": {"user_message": "hi"}})
    assert resp.status_code == 200
    assert resp.json()["data"]["agent_response"] == "hello"


def test_get_workflow_returns_v2_graph():
    client = make_client_app()
    resp = client.get("/workflow")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "srv"
    assert any(n["type"] == "llm" for n in body["nodes"])


def test_liveness():
    client = make_client_app()
    assert client.get("/liveness").json() == {"status": "ok"}


def test_stream_endpoint_removed():
    client = make_client_app()
    resp = client.post("/stream_agent", json={"data": {"user_message": "hi"}})
    assert resp.status_code == 404
