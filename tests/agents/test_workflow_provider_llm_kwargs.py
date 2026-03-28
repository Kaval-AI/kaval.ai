import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel
from kavalai.agents.workflow import Workflow
from kavalai.agents.workflow_model import WorkflowModel, LLMTask, AgentTask
from kavalai.agents.run_context import RunContext
from kavalai.agents.db import ModelCallStat


class MockOutput(BaseModel):
    result: str


@pytest.mark.asyncio
async def test_workflow_openai_llm_kwargs():
    """
    Test that OpenAI-specific llm_kwargs are passed to the OpenAI client.
    """
    task = LLMTask(
        name="openai_task",
        prompt="hello",
        output="output_type",
        llm_kwargs={
            "presence_penalty": 0.5,
            "frequency_penalty": 0.2,
            "reasoning_effort": "medium",
        },
    )

    model = WorkflowModel(
        name="openai_wf",
        data_types={
            "output_type": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
            }
        },
        tasks=[task],
        llm_model="openai/gpt-4o",
    )

    # Use a dummy stats object to return from the mock
    mock_stats = ModelCallStat(
        call_type="llm",
        model="openai/gpt-4o",
        duration_seconds=0.1,
        prompt_tokens=10,
        completion_tokens=20,
        response_data={"result": "success"},
    )

    with patch("kavalai.llm_clients.openai_client.AsyncOpenAI") as MockAsyncOpenAI:
        # Mock the responses.stream context manager
        mock_client = MockAsyncOpenAI.return_value
        _ = mock_client.responses

        # We need to mock the async context manager responses.stream
        mock_stream = MagicMock()
        mock_stream.__aenter__ = AsyncMock()
        mock_stream.__aexit__ = AsyncMock()

        # The mock_stream.__aenter__ returns an async iterator (the stream itself)
        # For simplicity, we can mock the OpenAIClient directly instead of AsyncOpenAI
        # because OpenAIClient.chat_completions is where the filtering and call happens.

    with patch("kavalai.agents.workflow.LLMClient") as MockLLMClient:
        mock_llm_instance = MockLLMClient.return_value
        mock_llm_instance.chat_completions = AsyncMock(
            return_value=(MockOutput(result="success"), mock_stats)
        )

        wf = Workflow(model)
        wf.models = {"output_type": MockOutput}

        run_context = RunContext()
        await wf.run_llm_task(task, run_context, None)

        # Verify LLMClient was initialized with the correct model
        MockLLMClient.assert_called_with(model="openai/gpt-4o")

        # Verify chat_completions was called with the merged llm_kwargs
        args, kwargs = mock_llm_instance.chat_completions.call_args
        assert kwargs["presence_penalty"] == 0.5
        assert kwargs["frequency_penalty"] == 0.2
        assert kwargs["reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_workflow_gemini_llm_kwargs():
    """
    Test that Gemini-specific llm_kwargs are passed to the Gemini client.
    """
    task = LLMTask(
        name="gemini_task",
        prompt="hello",
        output="output_type",
        llm_kwargs={"top_k": 40, "top_p": 0.95, "stop_sequences": ["END"]},
    )

    model = WorkflowModel(
        name="gemini_wf",
        data_types={
            "output_type": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
            }
        },
        tasks=[task],
        llm_model="gemini/gemini-2.0-flash",
    )

    mock_stats = ModelCallStat(
        call_type="llm",
        model="gemini/gemini-2.0-flash",
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

        # Verify LLMClient was initialized with the correct model
        MockLLMClient.assert_called_with(model="gemini/gemini-2.0-flash")

        # Verify chat_completions was called with the merged llm_kwargs
        args, kwargs = mock_llm_instance.chat_completions.call_args
        assert kwargs["top_k"] == 40
        assert kwargs["top_p"] == 0.95
        assert kwargs["stop_sequences"] == ["END"]


@pytest.mark.asyncio
async def test_workflow_agent_llm_kwargs_passing():
    """
    Test that llm_kwargs from AgentTask are passed to the PlanningAgent.
    """
    task = AgentTask(
        name="agent_task",
        output="output_type",
        llm_kwargs={"custom_agent_param": "agent_val"},
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
        llm_model="openai/gpt-4o",
    )

    with patch("kavalai.agents.workflow.PlanningAgent") as MockPlanningAgent:
        mock_agent_instance = MockPlanningAgent.return_value
        mock_agent_instance.run = AsyncMock(return_value=MockOutput(result="success"))

        wf = Workflow(model)
        wf.models = {"output_type": MockOutput}

        run_context = RunContext()
        await wf.run_agent_task(task, run_context, None)

        # Verify PlanningAgent was initialized with the merged llm_kwargs
        args, kwargs = MockPlanningAgent.call_args
        assert kwargs["llm_kwargs"]["custom_agent_param"] == "agent_val"
