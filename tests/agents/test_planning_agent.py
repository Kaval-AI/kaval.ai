import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from kavalai.agents.planning_agent import PlanningAgent
from kavalai.llm_clients.llm_client import LLMClient
from kavalai.agents.workflow_model import AgentTask, WorkflowModel, WorkflowException
from kavalai.agents.run_context import RunContext
from kavalai.agents.db import ModelCallStat


@pytest.mark.asyncio
async def test_planning_agent_loop_limit():
    # Mock workflow and its dependencies
    workflow = MagicMock()
    workflow.workflow_model = WorkflowModel(
        name="test",
        llm_model="openai/gpt-4",
        data_types={
            "input": {},
            "output": {"type": "object", "properties": {"res": {"type": "string"}}},
        },
        tasks=[],
    )
    workflow.agent_service = None

    # Mock chat_completions to always return a tool call
    mock_stats = ModelCallStat(call_type="llm", model="test", duration_seconds=0.1)

    with patch.object(
        LLMClient, "chat_completions", new_callable=AsyncMock
    ) as mock_chat:
        from kavalai.agents.planning_agent import _ToolDirective

        mock_chat.return_value = (
            _ToolDirective(
                action="tool",
                tool_kind="rest",
                server="s1",
                name="t1",
                arguments={},
                result_key="rk",
            ),
            mock_stats,
        )

        agent = PlanningAgent(workflow)
        task = AgentTask(name="test_task", max_steps=3, output="output")
        run_context = RunContext(data={})

        # We also need to mock workflow.run_rest_tool
        workflow.run_rest_tool = AsyncMock()

        await agent.run(task, run_context, None)

        # Should have called LLM 3 times (max_steps)
        assert mock_chat.call_count == 3


@pytest.mark.asyncio
async def test_planning_agent_finish():
    workflow = MagicMock()
    workflow.workflow_model = WorkflowModel(
        name="test",
        llm_model="openai/gpt-4",
        data_types={
            "output": {"type": "object", "properties": {"final": {"type": "string"}}}
        },
        tasks=[],
    )
    workflow.agent_service = None

    mock_stats = ModelCallStat(call_type="llm", model="test", duration_seconds=0.1)

    with patch.object(
        LLMClient, "chat_completions", new_callable=AsyncMock
    ) as mock_chat:
        from kavalai.agents.planning_agent import _ToolDirective

        mock_chat.return_value = (
            _ToolDirective(action="finish", final_output_key="final_result"),
            mock_stats,
        )

        agent = PlanningAgent(workflow)
        task = AgentTask(name="test_task", max_steps=5, output="output")
        # Pre-set the final result in context
        run_context = RunContext(data={"final_result": "done"})

        await agent.run(task, run_context, None)

        assert mock_chat.call_count == 1
        assert run_context.data["output"] == "done"


@pytest.mark.asyncio
async def test_planning_agent_mcp_restricted():
    workflow = MagicMock()
    workflow.workflow_model = WorkflowModel(
        name="test",
        llm_model="openai/gpt-4",
        data_types={
            "output": {"type": "object", "properties": {"res": {"type": "string"}}}
        },
        tasks=[],
    )
    workflow.agent_service = None

    mock_stats = ModelCallStat(call_type="llm", model="test", duration_seconds=0.1)

    with patch.object(
        LLMClient, "chat_completions", new_callable=AsyncMock
    ) as mock_chat:
        from kavalai.agents.planning_agent import _ToolDirective

        mock_chat.return_value = (
            _ToolDirective(
                action="tool", tool_kind="mcp", server="forbidden_server", name="t1"
            ),
            mock_stats,
        )

        agent = PlanningAgent(workflow)
        task = AgentTask(
            name="test_task", max_steps=5, allowed_mcp_servers=["allowed_server"]
        )
        run_context = RunContext(data={})

        with pytest.raises(WorkflowException) as exc:
            await agent.run(task, run_context, None)
        assert "not allowed" in str(exc.value)


@pytest.mark.asyncio
async def test_planning_agent_timeout():
    workflow = MagicMock()
    workflow.workflow_model = WorkflowModel(
        name="test",
        llm_model="openai/gpt-4",
        data_types={
            "output": {"type": "object", "properties": {"res": {"type": "string"}}}
        },
        tasks=[],
    )
    workflow.agent_service = None

    async def slow_chat(*args, **kwargs):
        await asyncio.sleep(2)
        return None, None

    with patch.object(LLMClient, "chat_completions", side_effect=slow_chat):
        agent = PlanningAgent(workflow)
        task = AgentTask(name="test_task", max_steps=5, timeout=1)
        run_context = RunContext(data={})

        with pytest.raises(WorkflowException) as exc:
            await agent.run(task, run_context, None)
        assert "timed out" in str(exc.value)
