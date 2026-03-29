import pytest
from unittest.mock import AsyncMock, MagicMock
from kavalai.agents.workflow import Workflow
from kavalai.agents.workflow_model import WorkflowModel, RagQueryTask
from kavalai.agents.run_context import RunContext
from kavalai.agents.agent_service import AgentService
from kavalai.agents.rag_service import RagServiceResult


@pytest.mark.asyncio
async def test_run_rag_task(agents_session_maker, monkeypatch):
    # 1. Setup Mock for RagService.query
    mock_results = [
        RagServiceResult(
            id="550e8400-e29b-41d4-a716-446655440000",
            model="test-model",
            collection_name="test-collection",
            source_id="source-1",
            content="test content",
            embedding_size=1536,
            rag_metadata={"key": "value"},
            similarity=0.9,
        )
    ]

    mock_query = AsyncMock(return_value=mock_results)
    monkeypatch.setattr("kavalai.agents.workflow.RagService.query", mock_query)
    mock_rag_service = AsyncMock(query=mock_query)
    mock_from_session_maker = MagicMock(return_value=mock_rag_service)
    monkeypatch.setattr(
        "kavalai.agents.workflow.RagService.from_session_maker", mock_from_session_maker
    )

    # 2. Define Workflow and Task
    task = RagQueryTask(
        name="rag_task",
        text="search query",
        top_k=3,
        collection_name="test-collection",
        output="output",
    )

    workflow_model = WorkflowModel(
        name="test_workflow",
        llm_model="openai/gpt-4o",
        embedding_model="openai/test-embedding-model",
        data_types={
            "input": {"type": "object", "properties": {}},
            "output": {"type": "object", "properties": {}},
        },
        tasks=[task],
    )

    agent_service = AgentService(agents_session_maker)
    workflow = Workflow(workflow_model, agent_service=agent_service)

    run_context = RunContext(agent_service=agent_service)
    run_context.data = {"input": {}}

    # 3. Run the task
    await workflow.run_rag_task(task, run_context)

    # 4. Assertions
    assert "rag_task" in run_context.data
    assert len(run_context.data["rag_task"]) == 1
    assert run_context.data["rag_task"][0]["content"] == "test content"
    assert run_context.data["rag_task"][0]["similarity"] == 0.9
    assert run_context.data["rag_task"][0]["source_id"] == "source-1"
    assert run_context.data["rag_task"][0]["metadata"] == '{"key": "value"}'
    # Ensure other fields are NOT present
    assert "id" not in run_context.data["rag_task"][0]
    assert "model" not in run_context.data["rag_task"][0]

    # Verify RagService was initialized with the correct model
    mock_from_session_maker.assert_called_once_with(
        agent_service.session_maker, "openai/test-embedding-model"
    )

    mock_query.assert_called_once_with(
        text="search query",
        top_k=3,
        collection_name="test-collection",
        source_ids=None,
        keep_best=False,
    )

    # Verify TaskLogger.log_rag_query was called
    mock_log_rag_query = AsyncMock()
    monkeypatch.setattr(workflow.task_logger, "log_rag_query", mock_log_rag_query)

    # Rerunning with the mock
    run_context.run_id = "test-run-id"
    run_context.agent_id = "test-agent-id"
    run_context.session_id = "test-session-id"
    await workflow.run_rag_task(task, run_context)

    mock_log_rag_query.assert_called_once()


@pytest.mark.asyncio
async def test_run_rag_task_with_context_resolution(agents_session_maker, monkeypatch):
    mock_results = []
    mock_query = AsyncMock(return_value=mock_results)
    monkeypatch.setattr("kavalai.agents.workflow.RagService.query", mock_query)

    task = RagQueryTask(name="rag_task", text="input.query", top_k=5, output="output")

    workflow_model = WorkflowModel(
        name="test_workflow",
        data_types={
            "input": {"type": "object", "properties": {}},
            "output": {"type": "object", "properties": {}},
        },
        tasks=[task],
    )

    agent_service = AgentService(agents_session_maker)
    workflow = Workflow(workflow_model, agent_service=agent_service)

    run_context = RunContext(agent_service=agent_service)
    run_context.data = {"input": {"query": "resolved query"}}

    await workflow.run_rag_task(task, run_context)

    mock_query.assert_called_once()
    args, kwargs = mock_query.call_args
    assert kwargs["text"] == "resolved query"


@pytest.mark.asyncio
async def test_run_rag_task_empty_input(agents_session_maker, monkeypatch):
    task = RagQueryTask(name="rag_task", text="", output="output")

    workflow_model = WorkflowModel(
        name="test_workflow",
        data_types={
            "input": {"type": "object", "properties": {}},
            "output": {"type": "object", "properties": {}},
        },
        tasks=[task],
    )

    agent_service = AgentService(agents_session_maker)
    workflow = Workflow(workflow_model, agent_service=agent_service)

    run_context = RunContext(agent_service=agent_service)
    run_context.data = {"input": {}}

    await workflow.run_rag_task(task, run_context)

    assert "rag_task" in run_context.data
    assert run_context.data["rag_task"] == []


@pytest.mark.asyncio
async def test_run_rag_task_resolved_empty_input(agents_session_maker, monkeypatch):
    task = RagQueryTask(name="rag_task", text="input.query", output="output")

    workflow_model = WorkflowModel(
        name="test_workflow",
        data_types={
            "input": {"type": "object", "properties": {}},
            "output": {"type": "object", "properties": {}},
        },
        tasks=[task],
    )

    agent_service = AgentService(agents_session_maker)
    workflow = Workflow(workflow_model, agent_service=agent_service)

    run_context = RunContext(agent_service=agent_service)
    run_context.data = {"input": {"query": ""}}

    await workflow.run_rag_task(task, run_context)

    assert "rag_task" in run_context.data
    assert run_context.data["rag_task"] == []


@pytest.mark.asyncio
async def test_workflow_dispatch_to_rag_task(agents_session_maker, monkeypatch):
    # Mock run_rag_task to verify it's called during workflow execution
    mock_run_rag_task = AsyncMock()
    monkeypatch.setattr(Workflow, "run_rag_task", mock_run_rag_task)

    task = RagQueryTask(name="rag_task", text="query", output="output")

    workflow_model = WorkflowModel(
        name="test_workflow",
        data_types={
            "input": {
                "type": "object",
                "properties": {"user_message": {"type": "string"}},
            },
            "output": {"type": "object", "properties": {}},
        },
        tasks=[task],
    )

    agent_service = AgentService(agents_session_maker)
    workflow = Workflow(workflow_model, agent_service=agent_service)

    await workflow.run(input_data={"user_message": "hello"})

    mock_run_rag_task.assert_called_once()
