import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from kavalai.agents.workflow import Workflow
from kavalai.agents.workflow_model import WorkflowModel
from kavalai.agents.run_context import RunContext
from kavalai.llm_clients.common import Streamer


@pytest.mark.asyncio
async def test_run_planning_agent_streamer_init_stream_output():
    # Test that streamer is initialized when stream_output is True
    workflow_data = {
        "name": "test_workflow",
        "data_types": {"output": {"type": "object"}},
        "tasks": [
            {
                "name": "agent_task",
                "type": "agent",
                "prompt": "test prompt",
                "output": "output",
                "stream_output": True,
            }
        ],
    }
    workflow_model = WorkflowModel(**workflow_data)
    workflow = Workflow(workflow_model)
    task = workflow_model.tasks[0]
    run_context = RunContext()
    queue = asyncio.Queue()

    with patch("kavalai.agents.workflow.PlanningAgent") as MockPlanningAgent:
        # We don't want to actually run the agent, just check if it was initialized with a streamer
        MockPlanningAgent.return_value.run = AsyncMock(return_value={})

        await workflow.run_agent_task(task, run_context, queue)

        # Check that PlanningAgent was called and the streamer argument was not None
        args, kwargs = MockPlanningAgent.call_args
        assert kwargs["streamer"] is not None
        assert isinstance(kwargs["streamer"], Streamer)
        assert kwargs["streamer"].name == "output"


@pytest.mark.asyncio
async def test_run_planning_agent_streamer_init_stream_persisted():
    # Test that streamer is initialized when stream_persisted is True
    workflow_data = {
        "name": "test_workflow",
        "data_types": {"output": {"type": "object"}},
        "tasks": [
            {
                "name": "agent_task",
                "type": "agent",
                "prompt": "test prompt",
                "output": "output",
                "stream_persisted": True,
            }
        ],
    }
    workflow_model = WorkflowModel(**workflow_data)
    workflow = Workflow(workflow_model)
    task = workflow_model.tasks[0]
    run_context = RunContext()
    queue = asyncio.Queue()

    with patch("kavalai.agents.workflow.PlanningAgent") as MockPlanningAgent:
        MockPlanningAgent.return_value.run = AsyncMock(return_value={})
        await workflow.run_agent_task(task, run_context, queue)

        args, kwargs = MockPlanningAgent.call_args
        assert kwargs["streamer"] is not None
        assert kwargs["streamer"].name == "output"


@pytest.mark.asyncio
async def test_run_planning_agent_streamer_init_stream_updates():
    # Test that streamer is initialized when stream_updates is True
    workflow_data = {
        "name": "test_workflow",
        "data_types": {"output": {"type": "object"}},
        "tasks": [
            {
                "name": "agent_task",
                "type": "agent",
                "prompt": "test prompt",
                "output": "output",
                "stream_updates": True,
            }
        ],
    }
    workflow_model = WorkflowModel(**workflow_data)
    workflow = Workflow(workflow_model)
    task = workflow_model.tasks[0]
    run_context = RunContext()
    queue = asyncio.Queue()

    with patch("kavalai.agents.workflow.PlanningAgent") as MockPlanningAgent:
        MockPlanningAgent.return_value.run = AsyncMock(return_value={})
        await workflow.run_agent_task(task, run_context, queue)

        args, kwargs = MockPlanningAgent.call_args
        assert kwargs["streamer"] is not None
        assert kwargs["streamer"].name == "output"


@pytest.mark.asyncio
async def test_run_planning_agent_streamer_no_init():
    # Test that streamer is NOT initialized when all streaming flags are False
    workflow_data = {
        "name": "test_workflow",
        "data_types": {"output": {"type": "object"}},
        "tasks": [
            {
                "name": "agent_task",
                "type": "agent",
                "prompt": "test prompt",
                "output": "output",
                "stream_output": False,
                "stream_updates": False,
                "stream_persisted": False,
            }
        ],
    }
    workflow_model = WorkflowModel(**workflow_data)
    workflow = Workflow(workflow_model)
    task = workflow_model.tasks[0]
    run_context = RunContext()
    queue = asyncio.Queue()

    with patch("kavalai.agents.workflow.PlanningAgent") as MockPlanningAgent:
        MockPlanningAgent.return_value.run = AsyncMock(return_value={})
        await workflow.run_agent_task(task, run_context, queue)

        args, kwargs = MockPlanningAgent.call_args
        assert kwargs["streamer"] is None


@pytest.mark.asyncio
async def test_run_planning_agent_streamer_no_queue():
    # Test that streamer is NOT initialized when queue is None even if flags are True
    workflow_data = {
        "name": "test_workflow",
        "data_types": {"output": {"type": "object"}},
        "tasks": [
            {
                "name": "agent_task",
                "type": "agent",
                "prompt": "test prompt",
                "output": "output",
                "stream_output": True,
            }
        ],
    }
    workflow_model = WorkflowModel(**workflow_data)
    workflow = Workflow(workflow_model)
    task = workflow_model.tasks[0]
    run_context = RunContext()
    queue = None

    with patch("kavalai.agents.workflow.PlanningAgent") as MockPlanningAgent:
        MockPlanningAgent.return_value.run = AsyncMock(return_value={})
        await workflow.run_agent_task(task, run_context, queue)

        args, kwargs = MockPlanningAgent.call_args
        assert kwargs["streamer"] is None
