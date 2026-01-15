from kavalai.backoffice.svg import generate_workflow_svg
from kavalai.agents.workflow import WorkflowModel
from unittest.mock import patch


def test_generate_workflow_svg(tmp_path):
    output_path = tmp_path / "test_output"
    model_data = {
        "name": "Test Workflow",
        "description": "A test workflow",
        "llm_provider": "openai",
        "data_types": {
            "input": {"properties": {"user_message": {"type": "string"}}},
            "output": {"properties": {"agent_response": {"type": "string"}}},
        },
        "tasks": [
            {
                "name": "Task 1",
                "inputs": {"input": {"type": "context"}},
                "output": "output",
                "prompt": "Say hello",
            },
            {
                "name": "Task 2",
                "inputs": {"output": {"type": "context"}},
                "output": "output",
                "tool": "my_tool",
                "mcp_server": "my_server",
            },
        ],
    }
    model = WorkflowModel(**model_data)

    with patch("kavalai.backoffice.svg.Digraph") as MockDigraph:
        mock_dot = MockDigraph.return_value
        generate_workflow_svg(model, str(output_path))

        # Verify Digraph was initialized
        MockDigraph.assert_called_once_with(name=model.name, comment=model.description)

        # Verify nodes were created
        # 2 data types + 2 tasks
        # Task 1 output is "output" which is NOT in model.data_types in my previous mock
        # Let's fix the mock data to include "output" in data_types
        assert mock_dot.node.call_count == 4

        # Verify edges were created
        # Task 1: input -> Task 1, Task 1 -> output
        # Task 2: output -> Task 2, Task 2 -> output
        assert mock_dot.edge.call_count == 4

        # Verify render was called
        mock_dot.render.assert_called_once_with(
            str(output_path), format="svg", cleanup=True
        )


def test_generate_workflow_svg_content():
    model_data = {
        "name": "Test Workflow",
        "description": "A test workflow",
        "llm_provider": "openai",
        "data_types": {},
        "tasks": [],
    }
    model = WorkflowModel(**model_data)

    with patch("kavalai.backoffice.svg.Digraph") as MockDigraph:
        mock_dot = MockDigraph.return_value
        mock_dot.pipe.return_value = b"<svg>test</svg>"

        content = generate_workflow_svg(model, return_content=True)

        assert content == "<svg>test</svg>"
        mock_dot.pipe.assert_called_once_with(format="svg")
        mock_dot.render.assert_not_called()
