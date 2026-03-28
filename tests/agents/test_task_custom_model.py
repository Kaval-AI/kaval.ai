import pytest
from unittest.mock import AsyncMock, patch
from pydantic import BaseModel
from kavalai.agents.workflow import Workflow
from kavalai.agents.workflow_model import WorkflowModel, LLMTask, AgentTask
from kavalai.agents.run_context import RunContext
from kavalai.agents.db import ModelCallStat


class MockOutput(BaseModel):
    result: str


@pytest.mark.asyncio
async def test_llm_task_custom_model():
    """Test that LLMTask can specify its own llm_model."""
    # This will fail until llm_model is added to LLMTask
    task = LLMTask(
        name="custom_model_task",
        prompt="hello",
        output="output_type",
        llm_model="custom/model-x",
    )

    model = WorkflowModel(
        name="custom_wf",
        data_types={
            "output_type": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
            }
        },
        tasks=[task],
        llm_model="default/model",
    )

    mock_stats = ModelCallStat(
        call_type="llm",
        model="custom/model-x",
        duration_seconds=0.1,
        prompt_tokens=10,
        completion_tokens=20,
        response_data={"result": "success"},
    )

    with patch("kavalai.agents.workflow.LLMClient") as MockLLMClient:
        mock_llm_instance = MockLLMClient.return_value
        mock_llm_instance.chat_completions = AsyncMock(
            return_value=(MockOutput(result="success"), mock_stats)
        )

        wf = Workflow(model)
        wf.models = {"output_type": MockOutput}

        run_context = RunContext()
        # The execution logic also needs update to prioritize task.llm_model
        await wf.run_llm_task(task, run_context, None)

        # Verify LLMClient was initialized with the task-specific model
        MockLLMClient.assert_called_with(model="custom/model-x")


@pytest.mark.asyncio
async def test_agent_task_custom_model():
    """Test that AgentTask can specify its own llm_model."""
    # This will fail until llm_model is added to AgentTask
    task = AgentTask(
        name="custom_agent_task", output="output_type", llm_model="custom/agent-model"
    )

    model = WorkflowModel(
        name="agent_wf",
        data_types={
            "output_type": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
            }
        },
        tasks=[task],
        llm_model="default/model",
    )

    with patch("kavalai.agents.workflow.LLMClient") as MockLLMClient:
        with patch("kavalai.agents.workflow.PlanningAgent") as MockPlanningAgent:
            mock_agent_instance = MockPlanningAgent.return_value
            mock_agent_instance.run = AsyncMock(
                return_value=MockOutput(result="success")
            )

            wf = Workflow(model)
            wf.models = {"output_type": MockOutput}

            run_context = RunContext()
            # The execution logic also needs update to prioritize task.llm_model
            await wf.run_agent_task(task, run_context, None)

            # Verify LLMClient was initialized with the task-specific model
            MockLLMClient.assert_called_with(model="custom/agent-model")


@pytest.mark.asyncio
async def test_llm_task_llm_kwargs_merging():
    """Test that llm_kwargs from task are merged with workflow llm_kwargs."""
    task = LLMTask(
        name="kwargs_task",
        prompt="hello",
        output="output_type",
        llm_kwargs={"task_param": "task_val"},
    )

    model = WorkflowModel(
        name="kwargs_wf",
        data_types={
            "output_type": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
            }
        },
        tasks=[task],
        llm_model="default/model",
        llm_kwargs={"wf_param": "wf_val"},
    )

    mock_stats = ModelCallStat(
        call_type="llm",
        model="default/model",
        duration_seconds=0.1,
        prompt_tokens=10,
        completion_tokens=20,
        response_data={"result": "success"},
    )

    with patch("kavalai.agents.workflow.LLMClient") as MockLLMClient:
        mock_llm_instance = MockLLMClient.return_value
        mock_llm_instance.chat_completions = AsyncMock(
            return_value=(MockOutput(result="success"), mock_stats)
        )

        wf = Workflow(model)
        wf.models = {"output_type": MockOutput}

        run_context = RunContext()
        await wf.run_llm_task(task, run_context, None)

        # Verify chat_completions was called with the merged llm_kwargs
        _, kwargs = mock_llm_instance.chat_completions.call_args
        assert kwargs["wf_param"] == "wf_val"
        assert kwargs["task_param"] == "task_val"


@pytest.mark.asyncio
async def test_agent_task_llm_kwargs_merging():
    """Test that llm_kwargs from agent task are merged with workflow llm_kwargs."""
    task = AgentTask(
        name="agent_kwargs_task",
        output="output_type",
        llm_kwargs={"task_param": "task_val"},
    )

    model = WorkflowModel(
        name="agent_kwargs_wf",
        data_types={
            "output_type": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
            }
        },
        tasks=[task],
        llm_model="default/model",
        llm_kwargs={"wf_param": "wf_val"},
    )

    with patch("kavalai.agents.workflow.LLMClient") as _:
        with patch("kavalai.agents.workflow.PlanningAgent") as MockPlanningAgent:
            mock_agent_instance = MockPlanningAgent.return_value
            mock_agent_instance.run = AsyncMock(
                return_value=MockOutput(result="success")
            )

            wf = Workflow(model)
            wf.models = {"output_type": MockOutput}

            run_context = RunContext()
            await wf.run_agent_task(task, run_context, None)

            # Verify PlanningAgent was initialized with the merged llm_kwargs
            _, kwargs = MockPlanningAgent.call_args
            assert kwargs["llm_kwargs"]["wf_param"] == "wf_val"
            assert kwargs["llm_kwargs"]["task_param"] == "task_val"
