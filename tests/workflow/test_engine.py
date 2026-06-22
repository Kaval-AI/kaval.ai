import typing

import pytest

from kavalai.workflow import (
    WorkflowEngine,
    InMemoryDataStorage,
    SqliteDataStorage,
    SqliteTaskLogger,
)
from kavalai.workflow.state import WorkflowState
from kavalai.agents.workflow_model import WorkflowException
from kavalai.functionkernel import pythontool
from kavalai.llm_clients.base_client import BaseLlmClient, ModelCallStat
from pydantic import BaseModel


# --------------------------------------------------------------------------- fakes
def _default_for(annotation):
    if annotation is str:
        return "x"
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    if annotation is bool:
        return False
    return None


def _build(model, value_map):
    kwargs = {}
    for name, field in model.model_fields.items():
        ann = field.annotation
        if name == "tool_calls":
            kwargs[name] = []
        elif name == "output":
            inner = [a for a in typing.get_args(ann) if a is not type(None)]
            kwargs[name] = _build(inner[0], value_map) if inner else None
        elif name == "instructions":
            kwargs[name] = "done"
        elif name in value_map:
            kwargs[name] = value_map[name]
        else:
            kwargs[name] = _default_for(ann)
    return model(**kwargs)


class FakeLLMClient(BaseLlmClient):
    """Deterministic client that fills response models from a value map and
    emits one ModelCallStat per call (so the StatsBridge path is exercised)."""

    def __init__(self, model, parameters=None, stats_receiver=None, value_map=None):
        super().__init__(parameters, stats_receiver)
        self.value_map = value_map or {}
        self.calls = []

    async def chat_completions(self, *, chat_history, response_model=None):
        self.calls.append(chat_history)
        if self.model_stats_receiver is not None:
            self.model_stats_receiver.receive_model_stats(
                ModelCallStat(
                    call_type="llm",
                    model="fake",
                    total_tokens=1,
                    duration_seconds=0.0,
                )
            )
        if response_model is None:
            return None
        return _build(response_model, self.value_map)


def make_factory(value_map=None, raises=False):
    created = []

    def factory(model, parameters=None, stats_receiver=None):
        if raises:
            client = _RaisingClient(model, parameters, stats_receiver)
        else:
            client = FakeLLMClient(
                model, parameters, stats_receiver, value_map=value_map
            )
        created.append(client)
        return client

    factory.created = created
    return factory


class _RaisingClient(BaseLlmClient):
    def __init__(self, model, parameters=None, stats_receiver=None):
        super().__init__(parameters, stats_receiver)

    async def chat_completions(self, *, chat_history, response_model=None):
        raise RuntimeError("llm boom")


# ------------------------------------------------------------------------ schemas
DATA_TYPES = {
    "input": {"type": "object", "properties": {"user_message": {"type": "string"}}},
    "classification": {
        "type": "object",
        "properties": {"intent": {"type": "string"}},
    },
    "output": {
        "type": "object",
        "properties": {"agent_response": {"type": "string"}},
    },
}


def graph_dict(nodes, **extra):
    return {
        "name": "wf",
        "description": "test workflow",
        "llm_model": "openai/fake",
        "data_types": dict(DATA_TYPES),
        "nodes": nodes,
        **extra,
    }


# --------------------------------------------------------------------------- tests
async def test_linear_llm_workflow_persists_everything():
    nodes = [
        {"name": "s", "type": "start", "next": "answer"},
        {
            "name": "answer",
            "type": "llm",
            "prompt": "respond to {{ context.input.user_message }}",
            "inputs": {"input": {"type": "context", "value": "input"}},
            "output": "output",
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    storage = SqliteDataStorage()
    tlog = SqliteTaskLogger()
    factory = make_factory({"agent_response": "hi there"})
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes), storage=storage, task_logger=tlog, client_factory=factory
    )
    state = await engine.run({"user_message": "hello"})

    assert state.status == "completed"
    assert state.trace == ["s", "answer", "e"]
    assert state.output_data == {"agent_response": "hi there"}

    # A short invocation id is assigned and token usage is aggregated.
    assert state.invocation_id and len(state.invocation_id) == 8
    assert state.token_usage == {
        "model_calls": 1,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 1,
    }

    # Checkpointed state is reloadable and matches (token usage persisted too).
    loaded = await storage.load_state(state.run_id)
    assert loaded.status == "completed"
    assert loaded.output_data == {"agent_response": "hi there"}
    assert loaded.token_usage == state.token_usage
    assert loaded.invocation_id == state.invocation_id

    # Chat history captured both turns.
    history = await storage.get_chat_history(state.session_id)
    assert [(m.role, m.content) for m in history] == [
        ("user", "hello"),
        ("assistant", "hi there"),
    ]

    # Node + model stats logged.
    await tlog.flush()
    conn = await tlog._connect()
    async with conn.execute("SELECT name, node_type FROM tasks") as cur:
        tasks = [tuple(r) for r in await cur.fetchall()]
    assert tasks == [("answer", "llm")]
    async with conn.execute("SELECT count(*) FROM model_call_stats") as cur:
        assert (await cur.fetchone())[0] == 1

    await storage.close()
    await tlog.close()


@pytest.mark.parametrize(
    "message,expected_branch",
    [("hi", "yes_node"), ("bye", "no_node")],
)
async def test_if_branch_routing(message, expected_branch):
    nodes = [
        {"name": "s", "type": "start", "next": "branch"},
        {
            "name": "branch",
            "type": "if",
            "condition": "input.user_message == 'hi'",
            "then": "yes_node",
            "else": "no_node",
        },
        {
            "name": "yes_node",
            "type": "llm",
            "prompt": "p",
            "output": "output",
            "next": "e",
        },
        {
            "name": "no_node",
            "type": "llm",
            "prompt": "p",
            "output": "output",
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes), client_factory=make_factory({"agent_response": "r"})
    )
    state = await engine.run({"user_message": message})
    assert expected_branch in state.trace
    assert state.status == "completed"


async def test_switch_routing_and_default():
    nodes = [
        {"name": "s", "type": "start", "next": "classify"},
        {
            "name": "classify",
            "type": "llm",
            "prompt": "p",
            "output": "classification",
            "next": "route",
        },
        {
            "name": "route",
            "type": "switch",
            "expr": "classification.intent",
            "cases": {"news": "news_node"},
            "default": "chat_node",
        },
        {
            "name": "news_node",
            "type": "llm",
            "prompt": "p",
            "output": "output",
            "next": "e",
        },
        {
            "name": "chat_node",
            "type": "llm",
            "prompt": "p",
            "output": "output",
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    # intent == news -> news_node
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes),
        client_factory=make_factory({"intent": "news", "agent_response": "r"}),
    )
    state = await engine.run({"user_message": "x"})
    assert "news_node" in state.trace and "chat_node" not in state.trace

    # unknown intent -> default chat_node
    engine2 = WorkflowEngine.from_dict(
        graph_dict(nodes),
        client_factory=make_factory({"intent": "weather", "agent_response": "r"}),
    )
    state2 = await engine2.run({"user_message": "x"})
    assert "chat_node" in state2.trace and "news_node" not in state2.trace


async def test_function_node_executes_tool():
    class Greeting(BaseModel):
        agent_response: str

    @pythontool
    def greet(name: str) -> Greeting:
        return Greeting(agent_response=f"hi {name}")

    nodes = [
        {"name": "s", "type": "start", "next": "call"},
        {
            "name": "call",
            "type": "function",
            "tool": "python://greet",
            "inputs": {"name": {"type": "literal", "value": "Sam"}},
            "output": "output",
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(graph_dict(nodes), client_factory=make_factory())
    engine.kernel.register_python_tool("greet", greet)
    state = await engine.run({"user_message": "x"})
    assert state.status == "completed"
    assert state.output_data == {"agent_response": "hi Sam"}
    assert state.trace == ["s", "call", "e"]


async def test_agent_node():
    nodes = [
        {"name": "s", "type": "start", "next": "do"},
        {
            "name": "do",
            "type": "agent",
            "prompt": "do the thing",
            "output": "output",
            "max_steps": 3,
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes),
        client_factory=make_factory({"agent_response": "agent did it"}),
    )
    state = await engine.run({"user_message": "x"})
    assert state.status == "completed"
    assert state.output_data == {"agent_response": "agent did it"}


async def test_invocation_id_is_unique_per_run_and_tokens_aggregate():
    nodes = [
        {"name": "s", "type": "start", "next": "a"},
        {"name": "a", "type": "llm", "prompt": "p", "output": "output", "next": "b"},
        {"name": "b", "type": "llm", "prompt": "p", "output": "output", "next": "e"},
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes), client_factory=make_factory({"agent_response": "r"})
    )
    s1 = await engine.run({"user_message": "x"})
    s2 = await engine.run({"user_message": "y"})

    # Two llm nodes -> two model calls aggregated per run.
    assert s1.token_usage["model_calls"] == 2
    assert s1.token_usage["total_tokens"] == 2
    # Each run gets its own id.
    assert s1.invocation_id != s2.invocation_id
    # Totals do not leak across runs.
    assert s2.token_usage["model_calls"] == 2


async def test_no_storage_or_logger_still_runs():
    nodes = [
        {"name": "s", "type": "start", "next": "answer"},
        {
            "name": "answer",
            "type": "llm",
            "prompt": "p",
            "output": "output",
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes), client_factory=make_factory({"agent_response": "r"})
    )
    state = await engine.run({"user_message": "x"})
    assert state.status == "completed"
    assert state.run_id is None  # no storage -> no ids


async def test_cycle_guard():
    # if-node that always loops back to itself never reaches the end.
    nodes = [
        {"name": "s", "type": "start", "next": "loop"},
        {
            "name": "loop",
            "type": "if",
            "condition": "True",
            "then": "loop",
            "else": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes), client_factory=make_factory(), max_node_visits=10
    )
    with pytest.raises(WorkflowException, match="max node visits"):
        await engine.run({"user_message": "x"})


async def test_switch_no_match_no_default_halts():
    nodes = [
        {"name": "s", "type": "start", "next": "route"},
        {
            "name": "route",
            "type": "switch",
            "expr": "input.user_message",
            "cases": {"hi": "e"},
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(graph_dict(nodes), client_factory=make_factory())
    with pytest.raises(WorkflowException, match="no next node"):
        await engine.run({"user_message": "bye"})


async def test_failure_marks_state_failed_and_persists():
    nodes = [
        {"name": "s", "type": "start", "next": "boom"},
        {
            "name": "boom",
            "type": "llm",
            "prompt": "p",
            "output": "output",
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    storage = SqliteDataStorage()
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes), storage=storage, client_factory=make_factory(raises=True)
    )
    with pytest.raises(WorkflowException, match="llm boom"):
        await engine.run({"user_message": "x"})

    # The failed state was checkpointed.
    conn = await storage._connect()
    async with conn.execute("SELECT context FROM runs LIMIT 1") as cur:
        row = await cur.fetchone()
    failed = WorkflowState.from_json(row["context"])
    assert failed.status == "failed"
    assert "llm boom" in failed.error
    await storage.close()


async def test_from_yaml_invalid_raises():
    bad_yaml = """
name: bad
data_types:
  input:
    type: object
nodes:
  - {name: s, type: start, next: ghost}
  - {name: e, type: end}
"""
    with pytest.raises(WorkflowException, match="validation failed"):
        WorkflowEngine.from_yaml(bad_yaml)


def test_resolve_model_missing_raises(monkeypatch):
    monkeypatch.delenv("KAVALAI_DEFAULT_LLM_MODEL", raising=False)
    nodes = [
        {"name": "s", "type": "start", "next": "n"},
        {"name": "n", "type": "llm", "prompt": "p", "output": "output", "next": "e"},
        {"name": "e", "type": "end", "output": "output"},
    ]
    g = dict(graph_dict(nodes))
    g.pop("llm_model")
    engine = WorkflowEngine.from_dict(g, client_factory=make_factory())
    with pytest.raises(WorkflowException, match="No LLM model configured"):
        engine._resolve_model(None)


async def test_use_history_includes_prior_messages():
    nodes = [
        {"name": "s", "type": "start", "next": "answer"},
        {
            "name": "answer",
            "type": "llm",
            "prompt": "p",
            "output": "output",
            "next": "e",
            "use_history": True,
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    storage = SqliteDataStorage()
    factory = make_factory({"agent_response": "r"})
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes), storage=storage, client_factory=factory
    )
    # Seed a session with a prior message, then run on the same session.
    handle = await storage.initialize_run(workflow_name="wf")
    await storage.add_chat_message(
        agent_id=handle.agent_id,
        session_id=handle.session_id,
        run_id=handle.run_id,
        role="user",
        content="earlier",
    )
    await engine.run({"user_message": "now"}, session_id=handle.session_id)

    # The LLM client received the seeded history in its chat_history.
    client = factory.created[0]
    contents = [m.content for m in client.calls[0].messages]
    assert any(c == "earlier" for c in contents)
    await storage.close()


# The exact workflow the standalone chat-playground.html builds, with the
# in-browser model swapped for the test fake (browser/... only runs in Pyodide).
CHAT_WORKFLOW_YAML = """
name: Browser chat
description: A friendly assistant that remembers the conversation.
llm_model: openai/fake

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

start: begin
nodes:
  - name: begin
    type: start
    next: reply
  - name: reply
    type: llm
    use_history: true
    prompt: >
      You are a warm, concise assistant chatting with a user. Answer their
      latest message and put your reply in the agent_response field.
    inputs:
      input:
        type: context
        value: input
    output: output
    next: done
  - name: done
    type: end
    output: output
"""


async def test_chat_workflow_remembers_history_in_memory():
    """End-to-end chat-playground scenario over the thread-free in-memory store:
    one InMemoryDataStorage and a stable session id, so each turn's LLM call
    sees the whole prior conversation (no seeding — the engine accumulates it).
    """
    storage = InMemoryDataStorage()
    factory = make_factory({"agent_response": "ack"})
    engine = WorkflowEngine.from_yaml(
        CHAT_WORKFLOW_YAML, storage=storage, client_factory=factory
    )
    session = "chat-1"

    for msg in ["hello", "what did I say?", "and before that?"]:
        state = await engine.run({"user_message": msg}, session_id=session)
        assert state.status == "completed"
        assert state.output_data == {"agent_response": "ack"}
        assert state.session_id == session

    # Turn 3's LLM call must include every earlier user + assistant message.
    third_call = factory.created[2].calls[0]
    contents = [m.content for m in third_call.messages]
    assert "hello" in contents
    assert "what did I say?" in contents
    assert contents.count("ack") == 2  # the two prior assistant replies

    # Persisted history holds all six messages, oldest first.
    history = await storage.get_chat_history(session)
    assert [(m.role, m.content) for m in history] == [
        ("user", "hello"),
        ("assistant", "ack"),
        ("user", "what did I say?"),
        ("assistant", "ack"),
        ("user", "and before that?"),
        ("assistant", "ack"),
    ]


def test_servers_and_unmarked_python_function_registration():
    # Covers rest/mcp server registration and the pythontool() wrap of an
    # undecorated function in WorkflowEngine.__init__.
    nodes = [
        {"name": "s", "type": "start", "next": "e"},
        {"name": "e", "type": "end", "output": "output"},
    ]
    g = graph_dict(
        nodes,
        rest_servers=[{"name": "api", "url": "http://localhost:9999"}],
        mcp_servers=[{"name": "m", "command": "echo"}],
        python_functions=[
            # clean_text is NOT decorated with @pythontool -> exercises the wrap.
            {"name": "clean_text", "path": "kavalai.agents.utils.clean_text"}
        ],
    )
    engine = WorkflowEngine.from_dict(g, client_factory=make_factory())
    assert "clean_text" in engine.kernel.python_tools
    assert "api" in engine.kernel.rest_servers
    assert "m" in engine.kernel.mcp_servers


def test_make_prompt_with_basemodel():
    from kavalai.workflow.engine import make_prompt

    class Payload(BaseModel):
        v: int

    text = make_prompt("base", {"p": Payload(v=1), "q": "lit"})
    assert "INPUT DATA:" in text
    assert '"v":1' in text  # BaseModel serialized as JSON
    assert "q:lit" in text


def test_get_data_type_none():
    nodes = [
        {"name": "s", "type": "start", "next": "e"},
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(graph_dict(nodes), client_factory=make_factory())
    assert engine.get_data_type(None) is None
    assert engine.get_data_type("output") is not None


def test_next_node_end_returns_none():
    from kavalai.agents.run_context import RunContext

    nodes = [
        {"name": "s", "type": "start", "next": "e"},
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(graph_dict(nodes), client_factory=make_factory())
    end_node = engine.node_map["e"]
    assert engine._next_node(end_node, RunContext()) is None


def test_from_yaml_and_from_yaml_path(tmp_path):
    yaml_text = """
name: yamlwf
description: y
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
  - {name: s, type: start, next: e}
  - {name: e, type: end, output: output}
"""
    engine = WorkflowEngine.from_yaml(yaml_text, client_factory=make_factory())
    assert engine.graph.name == "yamlwf"

    path = tmp_path / "wf.yaml"
    path.write_text(yaml_text)
    engine2 = WorkflowEngine.from_yaml_path(str(path), client_factory=make_factory())
    assert engine2.graph.name == "yamlwf"


def test_from_dict_invalid_raises():
    bad = {
        "name": "bad",
        "data_types": {"input": {"type": "object"}},
        "nodes": [
            {"name": "s", "type": "start", "next": "ghost"},
            {"name": "e", "type": "end"},
        ],
    }
    with pytest.raises(WorkflowException, match="validation failed"):
        WorkflowEngine.from_dict(bad)


async def test_rest_function_node_passes_method():
    from unittest.mock import AsyncMock

    nodes = [
        {"name": "s", "type": "start", "next": "call"},
        {
            "name": "call",
            "type": "function",
            "tool": "rest://api.do",
            "output": "output",
            "method": "post",
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    g = graph_dict(nodes, rest_servers=[{"name": "api", "url": "http://localhost"}])
    engine = WorkflowEngine.from_dict(g, client_factory=make_factory())
    out_model = engine.get_data_type("output")
    engine.kernel.call_tool = AsyncMock(return_value=out_model(agent_response="ok"))

    state = await engine.run({"user_message": "x"})
    assert state.status == "completed"
    # The REST method was forwarded to the kernel call.
    _, kwargs = engine.kernel.call_tool.call_args
    assert kwargs["method"] == "post"
    assert kwargs["tool_uri"] == "rest://api.do"
