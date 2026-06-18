"""
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Optional, Literal, Union, Annotated, Any

from pydantic import BaseModel, Field, model_validator

# Reuse the v1 building blocks for input wiring and server/tool declarations so
# the v2 graph shares the exact same YAML vocabulary for these pieces.
from kavalai.agents.workflow_model import (
    ArgumentInfo,
    RestServer,
    McpServer,
    PythonFunction,
    TemplateModel,
)


class BaseNode(BaseModel):
    """Common fields shared by every node in a workflow graph.

    A node is one vertex in the DAG/state-machine. ``name`` uniquely identifies
    the node and is the target referenced by transitions (``next``/``then``/
    ``else``/``cases``/``default``).
    """

    name: str


class StartNode(BaseNode):
    """Interaction start node.

    The caller hands an input to this node; execution begins here and proceeds
    to ``next``.
    """

    type: Literal["start"] = "start"
    next: str


class EndNode(BaseNode):
    """Interaction end node.

    Reaching an end node terminates the interaction. ``output`` names the
    context variable whose value is returned to the caller.
    """

    type: Literal["end"] = "end"
    output: str = "output"


class LLMNode(BaseNode):
    """Single LLM completion node.

    Resolves ``inputs`` from context, renders ``prompt`` and calls the LLM,
    storing the structured result in the ``output`` context variable, then
    transitions to ``next``.
    """

    type: Literal["llm"] = "llm"
    prompt: str
    inputs: dict[str, ArgumentInfo] = {}
    output: str
    next: str
    use_history: bool = True
    llm_model: Optional[str] = None
    llm_kwargs: dict[str, Any] = Field(default_factory=dict)
    stream_output: bool = False


class AgentNode(BaseNode):
    """Multi-step agent node.

    Runs the v2 :class:`~kavalai.agents.agent.Agent` loop (tool calling) up
    to ``max_steps`` and stores the final result in ``output``.
    """

    type: Literal["agent"] = "agent"
    prompt: str
    inputs: dict[str, ArgumentInfo] = {}
    output: str
    next: str
    allowed_tools: list[str] = Field(default_factory=list)
    max_steps: int = 10
    llm_model: Optional[str] = None
    llm_kwargs: dict[str, Any] = Field(default_factory=dict)


class FunctionNode(BaseNode):
    """Function-call node.

    Invokes a single tool via the :class:`~kavalai.functionkernel.FunctionKernel`
    (``python://`` / ``rest://`` / ``mcp://`` URIs) and stores the result in
    ``output``.
    """

    type: Literal["function"] = "function"
    tool: str
    inputs: dict[str, ArgumentInfo] = {}
    output: str
    next: str
    method: str = "get"


class IfNode(BaseNode):
    """Boolean branch node.

    Evaluates the ``condition`` string expression (e.g. ``state.count > 3``)
    against the run context and transitions to ``then`` when truthy, otherwise
    to ``else_`` (authored as ``else`` in YAML).
    """

    type: Literal["if"] = "if"
    condition: str
    then: str
    else_: str = Field(alias="else")

    model_config = {"populate_by_name": True}


class SwitchNode(BaseNode):
    """Multi-way branch node.

    Evaluates the ``expr`` string expression, stringifies the result and looks
    it up in ``cases``; falls back to ``default`` when no case matches.
    """

    type: Literal["switch"] = "switch"
    expr: str
    cases: dict[str, str] = {}
    default: Optional[str] = None


Node = Annotated[
    Union[
        StartNode,
        EndNode,
        LLMNode,
        AgentNode,
        FunctionNode,
        IfNode,
        SwitchNode,
    ],
    Field(discriminator="type"),
]


class WorkflowGraph(BaseModel):
    """A v2 workflow: a directed graph of nodes forming a state machine.

    name: str - workflow / agent name.
    description: str - human-readable description.
    version: str - schema version.
    llm_model: str - default LLM model (``provider/model``); nodes may override.
    llm_kwargs: dict - default LLM kwargs; nodes may override.
    data_types: dict - JSON-schema data type definitions (parsed by SchemaParser).
    nodes: list[Node] - the graph vertices.
    start: str - optional explicit start node name (otherwise the single
        ``start`` node is used).
    """

    name: str
    description: str = ""
    version: str = "2.0"
    llm_model: Optional[str] = None
    llm_kwargs: dict[str, Any] = Field(default_factory=dict)
    data_types: dict[str, dict]
    rest_servers: list[RestServer] = []
    mcp_servers: list[McpServer] = []
    templates: list[TemplateModel] = []
    python_functions: list[PythonFunction] = []
    nodes: list[Node]
    start: Optional[str] = None

    @model_validator(mode="after")
    def validate_graph(self) -> "WorkflowGraph":
        names = [n.name for n in self.nodes]
        if len(names) != len(set(names)):
            dupes = {n for n in names if names.count(n) > 1}
            raise ValueError(f"Duplicate node names: {sorted(dupes)}")
        name_set = set(names)

        start_nodes = [n for n in self.nodes if isinstance(n, StartNode)]
        end_nodes = [n for n in self.nodes if isinstance(n, EndNode)]
        if not start_nodes:
            raise ValueError("Workflow must define at least one 'start' node.")
        if not end_nodes:
            raise ValueError("Workflow must define at least one 'end' node.")

        # Resolve / validate the entry point.
        if self.start is not None:
            if self.start not in name_set:
                raise ValueError(f"start references unknown node '{self.start}'.")
        elif len(start_nodes) > 1:
            raise ValueError(
                "Multiple 'start' nodes found; set 'start' to choose the entry point."
            )
        else:
            self.start = start_nodes[0].name

        # Validate every transition target references an existing node.
        for node in self.nodes:
            for target in self._transition_targets(node):
                if target not in name_set:
                    raise ValueError(
                        f"Node '{node.name}' transitions to unknown node '{target}'."
                    )

        # Validate that node outputs are declared data types.
        for node in self.nodes:
            output = getattr(node, "output", None)
            if output is not None and output not in self.data_types:
                raise ValueError(
                    f"Node '{node.name}' output '{output}' is not declared in data_types."
                )

        return self

    @staticmethod
    def _transition_targets(node: "Node") -> list[str]:
        """Return the set of node names a node may transition to."""
        if isinstance(node, IfNode):
            return [node.then, node.else_]
        if isinstance(node, SwitchNode):
            targets = list(node.cases.values())
            if node.default is not None:
                targets.append(node.default)
            return targets
        if isinstance(node, EndNode):
            return []
        return [node.next]

    @property
    def node_map(self) -> dict[str, "Node"]:
        return {n.name: n for n in self.nodes}
