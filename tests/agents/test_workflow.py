import json
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4

import pytest

from kavalai.agents.workflow import Workflow, WorkflowModel, RunContext, make_prompt


@pytest.fixture
def wf():
    """Minimal workflow setup matching the updated WorkflowModel."""
    config = {
        "name": "TestAgent",
        "description": "A test workflow",
        "llm_provider": "openai/gpt-4o",  # Defined at top level in WorkflowModel
        "data_types": {
            "input": {
                "type": "object",
                "properties": {"user_message": {"type": "string"}},
            },
            "output": {
                "type": "object",
                "properties": {"agent_response": {"type": "string"}},
            },
            "data": {"type": "object", "properties": {"val": {"type": "string"}}},
        },
        "mcp_servers": [{"name": "srv", "url": "http://sse"}],
        "tasks": [
            {"name": "t1", "tool": "get_val", "mcp_server": "srv", "output": "data"},
            {
                "name": "t2",
                "prompt": "Translate this",
                "inputs": {
                    "data": {
                        "type": "context",
                        "name": "data",
                    }  # Updated to TypeInputInfo structure
                },
                "output": "output",
            },
        ],
    }
    return Workflow(WorkflowModel(**config))


@pytest.fixture
def ctx():
    """Updated RunContext using run_id."""
    return RunContext(
        agent_id=uuid4(),
        session_id=uuid4(),  # Renamed from interaction_id
        data={},
    )


@pytest.mark.asyncio
class TestWorkflowLogic:
    async def test_run_prompt(self, wf, ctx):
        """Verifies LLM task execution and context storage."""
        # Setup context with dependency data
        ctx.data["data"] = wf.get_data_type("data")(val="test-val")

        mock_res = wf.get_data_type("output")(agent_response="AI Response")

        with patch("instructor.from_provider") as mock_inst:
            # Setup the nested async mock for instructor
            mock_cmpl = AsyncMock()
            mock_cmpl.create.return_value = mock_res
            mock_inst.return_value.chat.completions = mock_cmpl

            await wf.run_prompt(wf.tasks["t2"], ctx)

            assert ctx.data["output"].agent_response == "AI Response"
            assert isinstance(ctx.data["output"], wf.get_data_type("output"))

    async def test_run_tool(self, wf, ctx):
        """Verifies MCP tool execution and JSON-to-Pydantic conversion."""
        # Mock the SSE and Session stack
        mock_mcp_content = MagicMock()
        mock_mcp_content.text = json.dumps({"val": "tool-val"})

        mock_mcp_res = MagicMock(content=[mock_mcp_content])

        mock_session = AsyncMock()
        mock_session.call_tool.return_value = mock_mcp_res

        with patch("kavalai.agents.workflow.sse_client") as mock_sse, patch(
            "kavalai.agents.workflow.ClientSession"
        ) as mock_client:
            # Setup async context managers
            mock_sse.return_value.__aenter__.return_value = (None, None)
            mock_client.return_value.__aenter__.return_value = mock_session

            await wf.run_tool(wf.tasks["t1"], ctx)

            assert ctx.data["data"].val == "tool-val"
            mock_session.call_tool.assert_called_once()

    async def test_make_prompt_integration(self, wf):
        """Checks if input data is correctly serialized into the system prompt."""
        data = {"my_key": wf.get_data_type("data")(val="hello")}
        prompt = make_prompt("Task:", data)

        assert "INPUT DATA:" in prompt
        # Verify JSON serialization of Pydantic models in prompt
        assert 'my_key:{"val":"hello"}' in prompt

    @pytest.mark.asyncio
    async def test_workflow_full_run_mocked(self, wf):
        """Test the full run method with service mocks."""
        mock_service = MagicMock()
        mock_service.get_agent_id.return_value = uuid4()
        mock_service.create_run.return_value = (
            uuid4()
        )  # Updated from create_interaction

        wf.agent_service = mock_service

        # Mock the individual task runners to avoid network calls
        with patch.object(wf, "run_prompt", new_callable=AsyncMock), patch.object(
            wf, "run_tool", new_callable=AsyncMock
        ):
            # Create a dummy output in context data so run() doesn't fail
            def side_effect(task, context):
                if task.output == "output":
                    context.data["output"] = wf.get_data_type("output")(
                        agent_response="Done"
                    )

            wf.run_prompt.side_effect = side_effect

            result = await wf.run(
                input_data={"user_message": "Hello"}, session_id=uuid4()
            )

            assert result.data.agent_response == "Done"
            assert result.session_id
            assert mock_service.add_message.called
