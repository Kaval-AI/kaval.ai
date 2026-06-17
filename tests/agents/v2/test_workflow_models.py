import pytest
import yaml
from pydantic import ValidationError

from kavalai.agents.v2.workflow.models import (
    EndNode,
    IfNode,
    StartNode,
    SwitchNode,
    WorkflowGraph,
)

BASE_DATA_TYPES = {
    "input": {"type": "object", "properties": {"user_message": {"type": "string"}}},
    "output": {"type": "object", "properties": {"agent_response": {"type": "string"}}},
}


def make_graph(nodes, **extra):
    return WorkflowGraph(
        name="wf", data_types=dict(BASE_DATA_TYPES), nodes=nodes, **extra
    )


def test_minimal_valid_graph():
    graph = make_graph(
        [
            {"name": "s", "type": "start", "next": "e"},
            {"name": "e", "type": "end", "output": "output"},
        ]
    )
    assert graph.start == "s"
    assert isinstance(graph.node_map["s"], StartNode)
    assert isinstance(graph.node_map["e"], EndNode)


def test_discriminated_node_parsing():
    graph = make_graph(
        [
            {"name": "s", "type": "start", "next": "branch"},
            {
                "name": "branch",
                "type": "if",
                "condition": "input.user_message == 'hi'",
                "then": "e",
                "else": "e",
            },
            {
                "name": "sw",
                "type": "switch",
                "expr": "input.user_message",
                "cases": {"hi": "e"},
                "default": "e",
            },
            {"name": "e", "type": "end"},
        ]
    )
    branch = graph.node_map["branch"]
    assert isinstance(branch, IfNode)
    assert branch.else_ == "e"  # 'else' alias maps to else_
    assert isinstance(graph.node_map["sw"], SwitchNode)


def test_missing_start_node():
    with pytest.raises(ValidationError, match="at least one 'start'"):
        make_graph([{"name": "e", "type": "end"}])


def test_missing_end_node():
    with pytest.raises(ValidationError, match="at least one 'end'"):
        make_graph([{"name": "s", "type": "start", "next": "s"}])


def test_duplicate_node_names():
    with pytest.raises(ValidationError, match="Duplicate node names"):
        make_graph(
            [
                {"name": "s", "type": "start", "next": "e"},
                {"name": "s", "type": "end"},
                {"name": "e", "type": "end"},
            ]
        )


def test_unknown_transition_target():
    with pytest.raises(ValidationError, match="unknown node 'nope'"):
        make_graph(
            [
                {"name": "s", "type": "start", "next": "nope"},
                {"name": "e", "type": "end"},
            ]
        )


def test_switch_unknown_case_target():
    with pytest.raises(ValidationError, match="unknown node"):
        make_graph(
            [
                {"name": "s", "type": "start", "next": "sw"},
                {
                    "name": "sw",
                    "type": "switch",
                    "expr": "input.user_message",
                    "cases": {"hi": "ghost"},
                },
                {"name": "e", "type": "end"},
            ]
        )


def test_undeclared_output_data_type():
    with pytest.raises(ValidationError, match="not declared in data_types"):
        make_graph(
            [
                {"name": "s", "type": "start", "next": "n"},
                {
                    "name": "n",
                    "type": "llm",
                    "prompt": "p",
                    "output": "undeclared",
                    "next": "e",
                },
                {"name": "e", "type": "end"},
            ]
        )


def test_multiple_start_requires_explicit_entry():
    nodes = [
        {"name": "s1", "type": "start", "next": "e"},
        {"name": "s2", "type": "start", "next": "e"},
        {"name": "e", "type": "end"},
    ]
    with pytest.raises(ValidationError, match="set 'start'"):
        make_graph(nodes)
    # Providing an explicit start resolves the ambiguity.
    graph = make_graph(nodes, start="s2")
    assert graph.start == "s2"


def test_explicit_start_unknown():
    with pytest.raises(ValidationError, match="start references unknown node"):
        make_graph(
            [
                {"name": "s", "type": "start", "next": "e"},
                {"name": "e", "type": "end"},
            ],
            start="ghost",
        )


def test_from_yaml_roundtrip():
    text = """
name: yaml_wf
description: built from yaml
llm_model: openai/x
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
    graph = WorkflowGraph(**yaml.safe_load(text))
    assert graph.name == "yaml_wf"
    assert graph.llm_model == "openai/x"
    assert graph.start == "s"
