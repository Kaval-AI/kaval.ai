from kavalai.backoffice.svg import generate_workflow_svg
from kavalai.agents.workflow_model import WorkflowModel
from unittest.mock import patch


def test_generate_workflow_svg(tmp_path):
    output_path = tmp_path / "test_output"
    model_data = {
        "name": "Test Workflow",
        "description": "A test workflow",
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
                "rest_server": "my_server",
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
        assert mock_dot.node.call_count == 4

        # Verify edges were created
        assert mock_dot.edge.call_count == 4

        # Verify render was called
        mock_dot.render.assert_called_once_with(
            str(output_path), format="svg", cleanup=True
        )


def test_generate_workflow_svg_with_ref():
    model_data = {
        "name": "Ref Workflow",
        "llm_model": "gpt-4o",
        "data_types": {
            "input": {"$ref": "CommonInput"},
            "CommonInput": {"properties": {"p": {"type": "string"}}},
            "output": {"properties": {"r": {"type": "string"}}},
        },
        "tasks": [
            {
                "name": "Task 1",
                "inputs": {"input": {"type": "context"}},
                "output": "output",
                "prompt": "Say hello",
            }
        ],
    }
    model = WorkflowModel(**model_data)

    with patch("kavalai.backoffice.svg.Digraph") as MockDigraph:
        mock_dot = MockDigraph.return_value
        generate_workflow_svg(model)

        # Check node calls
        node_calls = {
            call.args[0]: call.args[1] for call in mock_dot.node.call_args_list
        }
        assert "input" in node_calls
        assert node_calls["input"] == "input : CommonInput"
        assert "CommonInput" not in node_calls
        assert "output" in node_calls
        assert "task_Task_1" in node_calls


def test_generate_workflow_svg_content():
    model_data = {
        "name": "Test Workflow",
        "description": "A test workflow",
        "llm_model": "gpt-4o",
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


def test_generate_workflow_svg_task_with_tool():
    model_data = {
        "name": "Tool Workflow",
        "data_types": {
            "input": {"properties": {"p": {"type": "string"}}},
            "output": {"properties": {"r": {"type": "string"}}},
        },
        "tasks": [
            {
                "name": "Tool Task",
                "inputs": {"input": {"type": "context"}},
                "output": "output",
                "tool": "my_tool",
            }
        ],
    }
    model = WorkflowModel(**model_data)

    with patch("kavalai.backoffice.svg.Digraph") as MockDigraph:
        mock_dot = MockDigraph.return_value
        generate_workflow_svg(model)

        node_calls = {
            call.args[0]: call.args[1] for call in mock_dot.node.call_args_list
        }
        assert "task_Tool_Task" in node_calls
        assert "Tool Task\n(Tool: my_tool)" in node_calls["task_Tool_Task"]


def test_generate_workflow_svg_unreferenced_type():
    model_data = {
        "name": "Unreferenced Workflow",
        "data_types": {
            "input": {"properties": {"p": {"type": "string"}}},
            "output": {"properties": {"r": {"type": "string"}}},
            "Unused": {"properties": {"u": {"type": "string"}}},
        },
        "tasks": [
            {
                "name": "Task 1",
                "inputs": {"input": {"type": "context"}},
                "output": "output",
                "prompt": "Say hello",
            }
        ],
    }
    model = WorkflowModel(**model_data)

    with patch("kavalai.backoffice.svg.Digraph") as MockDigraph:
        mock_dot = MockDigraph.return_value
        generate_workflow_svg(model)

        node_calls = {
            call.args[0]: call.args[1] for call in mock_dot.node.call_args_list
        }
        assert "input" in node_calls
        assert "output" in node_calls
        assert "Unused" not in node_calls


def test_generate_workflow_svg_complex_paths():
    model_data = {
        "name": "Complex Workflow",
        "data_types": {
            "input": {
                "properties": {"user": {"properties": {"name": {"type": "string"}}}}
            },
            "intermediate": {"properties": {"val": {"type": "integer"}}},
            "output": {"properties": {"result": {"type": "string"}}},
        },
        "tasks": [
            {
                "name": "Task 1",
                "inputs": {
                    "user_name": {"value": "input.user.name", "type": "context"}
                },
                "output": "intermediate",
                "tool": "get_val",
            },
            {
                "name": "Task 2",
                "inputs": {"data": {"value": "intermediate", "type": "context"}},
                "output": {
                    "result": {"value": "Task 1.output", "type": "context"},
                    "final": {"value": "something.else", "type": "context"},
                },
            },
        ],
    }
    model = WorkflowModel(**model_data)

    with patch("kavalai.backoffice.svg.Digraph") as MockDigraph:
        mock_dot = MockDigraph.return_value
        generate_workflow_svg(model)

        node_calls = {
            call.args[0]: call.args[1] for call in mock_dot.node.call_args_list
        }
        # Data nodes
        assert "input" in node_calls
        assert "intermediate" in node_calls
        assert "output" in node_calls  # Combine task implicitly uses "output" type
        assert "something" in node_calls  # Referenced in task 2 output

        # Edges
        edge_sources = [call.args[0] for call in mock_dot.edge.call_args_list]
        edge_targets = [call.args[1] for call in mock_dot.edge.call_args_list]

        # Task 1: input -> Task 1, Task 1 -> intermediate
        assert "input" in edge_sources
        assert "task_Task_1" in edge_targets
        assert "task_Task_1" in edge_sources
        assert "intermediate" in edge_targets

        # Task 2: intermediate -> Task 2, Task 2 -> output, Task 1 -> Task 2, something -> Task 2
        assert "intermediate" in edge_sources
        assert "task_Task_2" in edge_targets
        assert "task_Task_2" in edge_sources
        assert "output" in edge_targets
        # Mapping context from other task
        assert "Task 1" in edge_sources
        assert "task_Task_2" in edge_targets
        # Mapping context from unknown but added type
        assert "something" in edge_sources
