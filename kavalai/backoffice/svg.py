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

from typing import List, Dict, Set
from graphviz import Digraph
from pydantic import BaseModel
from kavalai.agents.workflow import WorkflowModel


class Node(BaseModel):
    id: str
    label: str
    shape: str
    color: str
    style: str = "filled"


class Edge(BaseModel):
    source: str
    target: str


def generate_workflow_svg(
    model: WorkflowModel,
    output_filename: str = "workflow_graph",
    return_content: bool = False,
):
    nodes: Dict[str, Node] = {}
    edges: List[Edge] = []

    # 0. Identify used and base types
    used_types: Set[str] = set()
    # Always include "input" as it's the entry point
    if "input" in model.data_types:
        used_types.add("input")

    for task in model.tasks:
        if isinstance(task.output, str):
            used_types.add(task.output)
        elif isinstance(task.output, dict):
            # For combine tasks, if it's the final output, it uses the "output" data type
            # We also add all types referenced in the values
            used_types.add("output")
            for out_info in task.output.values():
                if out_info.type == "context":
                    actual_name = (out_info.name or out_info.value or "").split(".")[0]
                    if actual_name:
                        used_types.add(actual_name)

        for input_name, input_info in task.inputs.items():
            if input_info.type == "context":
                actual_name = (input_info.name or input_info.value or input_name).split(
                    "."
                )[0]
                used_types.add(actual_name)

    # Resolve $ref chains to find all types that should be mentioned in labels
    # Note: we don't add ref_type to used_types here because we only want to render
    # nodes that are directly used as inputs or outputs.

    # 1. Define Data Type Nodes
    # First, collect all potential node names from data_types and used_types
    all_node_names = set(model.data_types.keys()) | used_types

    for dt_name in all_node_names:
        dt_def = model.data_types.get(dt_name, {})
        # Only render types that are directly used as inputs or outputs
        is_directly_used = dt_name == "input" or dt_name in used_types
        if not is_directly_used:
            for task in model.tasks:
                if isinstance(task.output, str):
                    if task.output == dt_name:
                        is_directly_used = True
                        break
                elif isinstance(task.output, dict):
                    if dt_name == "output":
                        is_directly_used = True
                        break
                    for out_info in task.output.values():
                        if out_info.type == "context":
                            source_id = (out_info.name or out_info.value or "").split(
                                "."
                            )[0]
                            if source_id == dt_name:
                                is_directly_used = True
                                break
                    if is_directly_used:
                        break

                for input_name, input_info in task.inputs.items():
                    if input_info.type == "context":
                        source_id = (
                            input_info.name or input_info.value or input_name
                        ).split(".")[0]
                        if source_id == dt_name:
                            is_directly_used = True
                            break
                if is_directly_used:
                    break

        if not is_directly_used:
            continue

        label = dt_name
        if isinstance(dt_def, dict) and "$ref" in dt_def:
            label = f"{dt_name} : {dt_def['$ref']}"

        nodes[dt_name] = Node(
            id=dt_name,
            label=label,
            shape="box",
            color="lightblue2",
        )

    # 2. Define Task Nodes and Edges
    for task in model.tasks:
        task_id = f"task_{task.name.replace(' ', '_')}"
        label = (
            f"{task.name}\n(Tool: {task.tool})"
            if task.tool
            else f"{task.name}\n(LLM Prompt)"
        )

        nodes[task_id] = Node(
            id=task_id,
            label=label,
            shape="rect",
            style="filled,rounded",
            color="darkseagreen1",
        )

        # Edges: Inputs -> Task
        for input_name, input_info in task.inputs.items():
            if input_info.type == "context":
                source_id = (input_info.name or input_info.value or input_name).split(
                    "."
                )[0]
                if source_id in nodes:
                    edges.append(Edge(source=source_id, target=task_id))

        # Edges: Task -> Output
        if isinstance(task.output, str):
            if task.output in nodes:
                edges.append(Edge(source=task_id, target=task.output))
        elif isinstance(task.output, dict):
            if "output" in nodes:
                edges.append(Edge(source=task_id, target="output"))
            for out_info in task.output.values():
                if out_info.type == "context":
                    source_id = (out_info.name or out_info.value or "").split(".")[0]
                    if source_id in nodes:
                        edges.append(Edge(source=source_id, target=task_id))

    # 3. Render using Graphviz
    dot = Digraph(name=model.name, comment=model.description)
    dot.attr(rankdir="LR", size="10,10")
    dot.attr("node", fontname="Arial", fontsize="12")

    for node in nodes.values():
        dot.node(
            node.id,
            node.label,
            shape=node.shape,
            style=node.style,
            color=node.color,
        )

    for edge in edges:
        dot.edge(edge.source, edge.target)

    if return_content:
        return dot.pipe(format="svg").decode("utf-8")

    dot.render(output_filename, format="svg", cleanup=True)
    print(f"Workflow SVG generated: {output_filename}.svg")


# Example usage:
if __name__ == "__main__":
    import yaml

    with open("kavalai/agents/example.yaml", "r") as f:
        data = yaml.safe_load(f)
        model = WorkflowModel(**data)
        generate_workflow_svg(model)
