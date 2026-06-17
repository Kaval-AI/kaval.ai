import pytest

from kavalai.agents.v2.workflow import WorkflowBuilder, WorkflowEngine
from kavalai.agents.v2.workflow.builder import _coerce_inputs, _field_schema
from kavalai.agents.v2.workflow.models import (
    AgentNode,
    FunctionNode,
    IfNode,
    LLMNode,
    SwitchNode,
)
from kavalai.agents.workflow_model import ArgumentInfo


def _base(builder: WorkflowBuilder) -> WorkflowBuilder:
    return builder.data_type("input", {"user_message": str}).data_type(
        "output", {"agent_response": str}
    )


def test_field_schema_variants():
    assert _field_schema(str) == {"type": "string"}
    assert _field_schema(int) == {"type": "integer"}
    assert _field_schema("number") == {"type": "number"}
    raw = {"type": "array", "items": {"type": "string"}}
    assert _field_schema(raw) is raw


def test_field_schema_rejects_bad_type():
    with pytest.raises(TypeError):
        _field_schema(bytes)
    with pytest.raises(TypeError):
        _field_schema(123)


def test_coerce_inputs_variants():
    info = ArgumentInfo(type="literal", value="x")
    out = _coerce_inputs(
        {
            "a": "input.user_message",  # string -> context path
            "b": {"type": "literal", "value": 5},  # dict -> ArgumentInfo
            "c": info,  # ArgumentInfo passthrough
        }
    )
    assert out["a"] == ArgumentInfo(type="context", value="input.user_message")
    assert out["b"] == ArgumentInfo(type="literal", value=5)
    assert out["c"] is info
    assert _coerce_inputs(None) == {}


def test_coerce_inputs_rejects_bad_spec():
    with pytest.raises(TypeError):
        _coerce_inputs({"a": 123})


def test_build_minimal_graph():
    graph = (
        WorkflowBuilder("demo", description="d", llm_model="openai/x")
        .data_type("input", {"user_message": str})
        .data_type("output", {"agent_response": str})
        .start("reply")
        .llm(
            "reply", prompt="hi", inputs={"input": "input"}, output="output", next="end"
        )
        .end()
        .build()
    )
    assert graph.name == "demo"
    assert graph.llm_model == "openai/x"
    assert graph.start == "start"  # entry node (default name); its next is "reply"
    assert graph.node_map["start"].next == "reply"
    reply = graph.node_map["reply"]
    assert isinstance(reply, LLMNode)
    assert reply.inputs["input"] == ArgumentInfo(type="context", value="input")


def test_build_all_node_types():
    graph = (
        WorkflowBuilder("everything")
        .data_type("input", {"user_message": str})
        .data_type("flag", {"go": bool})
        .data_type("output", {"agent_response": str})
        .start("decide")
        .if_("decide", condition="True", then="sw", else_="act")
        .switch("sw", expr="input.user_message", cases={"x": "act"}, default="act")
        .agent("act", prompt="do", output="output", next="call", max_steps=2)
        .function("call", tool="python://noop", output="output", next="end")
        .end()
        .build()
    )
    assert isinstance(graph.node_map["decide"], IfNode)
    assert isinstance(graph.node_map["sw"], SwitchNode)
    assert isinstance(graph.node_map["act"], AgentNode)
    assert graph.node_map["act"].max_steps == 2
    assert isinstance(graph.node_map["call"], FunctionNode)
    assert graph.node_map["decide"].else_ == "act"


def test_data_type_schema_and_ref():
    graph = (
        WorkflowBuilder("dt")
        .data_type("input", {"user_message": str})
        .data_type(
            "feed",
            schema={"type": "object", "properties": {"url": {"type": "string"}}},
        )
        .data_type("output", ref="feed")
        .start("end")
        .end()
        .build()
    )
    assert graph.data_types["feed"]["properties"]["url"]["type"] == "string"
    assert graph.data_types["output"] == {"$ref": "feed"}


def test_resources_registered():
    builder = (
        _base(WorkflowBuilder("res"))
        .start("end")
        .end()
        .python_function("noop", "kavalai.agents.utils.to_plain")
        .rest_server({"name": "api", "url": "http://localhost"})
        .mcp_server({"name": "m", "command": "echo"})
        .template("greeting", "Hello {{ name }}")
    )
    graph = builder.build()
    assert graph.python_functions[0].name == "noop"
    assert graph.rest_servers[0].name == "api"
    assert graph.mcp_servers[0].name == "m"
    assert graph.templates[0].name == "greeting"


def test_explicit_node_name_overrides_default():
    graph = (
        _base(WorkflowBuilder("named"))
        .start("reply", name="begin")
        .llm("reply", prompt="hi", output="output", next="finish")
        .end(name="finish", output="output")
        .build()
    )
    assert graph.start == "begin"
    assert "finish" in graph.node_map


def test_build_engine_returns_engine():
    engine = (
        _base(WorkflowBuilder("eng", llm_model="openai/x"))
        .start("reply")
        .llm("reply", prompt="hi", output="output", next="end")
        .end()
        .build_engine()
    )
    assert isinstance(engine, WorkflowEngine)
    assert engine.graph.name == "eng"
