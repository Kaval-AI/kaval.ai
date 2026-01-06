import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from kavalai.agents.workflow import Workflow, WorkflowModel, RunContext, make_prompt


@pytest.fixture
def wf():
    """Minimal workflow setup."""
    config = {
        "name": "Test",
        "description": "...",
        "data_types": {
            "input": {"type": "object", "properties": {"msg": {"type": "string"}}},
            "output": {"type": "object", "properties": {"res": {"type": "string"}}},
            "data": {"type": "object", "properties": {"val": {"type": "string"}}},
        },
        "mcp_servers": [{"name": "srv", "url": "http://sse"}],
        "tasks": [
            {"name": "t1", "tool": "get", "mcp_server": "srv", "output": "data"},
            {
                "name": "t2",
                "prompt": "...",
                "llm_provider": "gpt",
                "inputs": ["data"],
                "output": "output",
            },
        ],
    }
    return Workflow(WorkflowModel(**config))


@pytest.fixture
def ctx():
    return RunContext()


@pytest.mark.asyncio
class TestWorkflowLogic:
    async def test_run_prompt(self, wf, ctx):
        """Verifies LLM task execution and context storage."""
        # Setup context with dependency data
        ctx.data["data"] = wf.get_data_type("data")(val="test-val")

        mock_res = wf.get_data_type("output")(res="AI Response")
        with patch("instructor.from_provider") as mock_inst:
            mock_inst.return_value.chat.completions.create = AsyncMock(
                return_value=mock_res
            )

            await wf.run_prompt(wf.tasks["t2"], ctx)

            assert ctx.data["output"].res == "AI Response"
            assert isinstance(ctx.data["output"], wf.get_data_type("output"))

    async def test_run_tool(self, wf, ctx):
        """Verifies MCP tool execution and JSON-to-Pydantic conversion."""
        # Mock the SSE and Session stack
        mock_mcp_res = MagicMock(
            content=[MagicMock(text=json.dumps({"val": "tool-val"}))]
        )
        mock_session = AsyncMock()
        mock_session.call_tool.return_value = mock_mcp_res

        with patch(
            "kavalai.agents.workflow.sse_client",
            return_value=MagicMock(__aenter__=AsyncMock(return_value=(None, None))),
        ), patch(
            "kavalai.agents.workflow.ClientSession",
            return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_session)),
        ):
            await wf.run_tool(wf.tasks["t1"], ctx)

            assert ctx.data["data"].val == "tool-val"
            mock_session.call_tool.assert_called_once()

    async def test_make_prompt_integration(self, wf, ctx):
        """Checks if input data is correctly serialized into the system prompt."""
        data = {"my_key": wf.get_data_type("data")(val="hello")}
        prompt = make_prompt("Task:", data)

        assert "INPUT DATA:" in prompt
        assert 'my_key:{"val":"hello"}' in prompt
