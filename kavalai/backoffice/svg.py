from graphviz import Digraph
from kavalai.agents.workflow import (
    WorkflowModel,
)  # Assuming your classes are in workflow.py


def generate_workflow_svg(
    model: WorkflowModel, output_filename: str = "workflow_graph"
):
    # Initialize the graph
    dot = Digraph(name=model.name, comment=model.description)
    dot.attr(rankdir="LR", size="10,10")

    # Global Styles
    dot.attr("node", fontname="Arial", fontsize="12")

    # 1. Define Data Type Nodes (Rectangles)
    for dt_name in model.data_types.keys():
        dot.node(dt_name, dt_name, shape="box", style="filled", color="lightblue2")

    # 2. Define Task Nodes (Rounded Rectangles)
    for task in model.tasks:
        # Create a unique ID for the task node to avoid name collisions
        task_id = f"task_{task.name.replace(' ', '_')}"

        # Determine label (Tool vs Prompt)
        label = (
            f"{task.name}\n(Tool: {task.tool})"
            if task.tool
            else f"{task.name}\n(LLM Prompt)"
        )

        dot.node(
            task_id, label, shape="rect", style="filled,rounded", color="darkseagreen1"
        )

        # 3. Create Edges
        # Inputs -> Task
        for input_name in task.inputs:
            if input_name in model.data_types:
                dot.edge(input_name, task_id)

        # Task -> Output
        if task.output in model.data_types:
            dot.edge(task_id, task.output)

    # Render the graph
    dot.render(output_filename, format="svg", cleanup=True)
    print(f"Workflow SVG generated: {output_filename}.svg")


# Example usage:
if __name__ == "__main__":
    import yaml

    with open("kavalai/agents/example.yaml", "r") as f:
        data = yaml.safe_load(f)
        model = WorkflowModel(**data)
        generate_workflow_svg(model)
