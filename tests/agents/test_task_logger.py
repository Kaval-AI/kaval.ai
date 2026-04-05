"""
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import asyncio

from kavalai.agents.task_logger import TaskLogger
from kavalai.agents.run_context import RunContext
from kavalai.agents.agent_service import AgentService


async def wait_for_background_tasks(task_logger: TaskLogger):
    """Helper to wait for all background logging tasks to complete."""
    if task_logger._background_tasks:
        await asyncio.gather(*task_logger._background_tasks, return_exceptions=True)


@pytest.fixture
def session_maker(agents_session_maker):
    return agents_session_maker


@pytest.fixture
def agent_service(session_maker):
    return AgentService(session_maker)


@pytest.mark.asyncio
async def test_task_logger_llm_task(agent_service, session_maker):
    """Test logging an LLM task."""
    # Create agent, session, and run
    agent = await agent_service.get_or_create_agent(
        name="test_agent",
        description="Test agent",
        input_schema={},
        output_schema={},
        workflow={},
    )
    session = await agent_service.get_or_create_session(agent_id=agent.id)
    run = await agent_service.create_run(session_id=session.id, input_data={})

    # Create run context
    run_context = RunContext(agent_service=agent_service)
    run_context.agent_id = agent.id
    run_context.session_id = session.id
    run_context.run_id = run.id

    # Create task logger and log an LLM task
    task_logger = TaskLogger(agent_service, run_context)
    task_logger.log_llm_task(
        task_name="test_llm_task",
        prompt="You are a helpful assistant. Answer: {{input.question}}",
        input_data={"question": "What is 2+2?"},
        output={"answer": "4"},
        duration=1.5,
    )
    await wait_for_background_tasks(task_logger)

    # Verify the task was logged
    async with session_maker() as db_session:
        from kavalai.agents.db import Task
        from sqlalchemy import select

        result = await db_session.execute(select(Task).where(Task.run_id == run.id))
        tasks = result.scalars().all()

        assert len(tasks) == 1
        task = tasks[0]
        assert task.name == "test_llm_task"
        assert task.prompt == "You are a helpful assistant. Answer: {{input.question}}"
        assert task.inputs == {"question": "What is 2+2?"}
        assert task.output == {"answer": "4"}
        assert task.duration_seconds == 1.5
        assert task.errors is None


@pytest.mark.asyncio
async def test_task_logger_llm_task_with_errors(agent_service, session_maker):
    """Test logging an LLM task with errors."""
    # Create agent, session, and run
    agent = await agent_service.get_or_create_agent(
        name="test_agent",
        description="Test agent",
        input_schema={},
        output_schema={},
        workflow={},
    )
    session = await agent_service.get_or_create_session(agent_id=agent.id)
    run = await agent_service.create_run(session_id=session.id, input_data={})

    # Create run context
    run_context = RunContext(agent_service=agent_service)
    run_context.agent_id = agent.id
    run_context.session_id = session.id
    run_context.run_id = run.id

    # Create task logger and log an LLM task with errors
    task_logger = TaskLogger(agent_service, run_context)
    task_logger.log_llm_task(
        task_name="test_llm_task_error",
        prompt="Test prompt",
        input_data={},
        output=None,
        duration=0.5,
        errors=["Validation error", "Timeout error"],
    )
    await wait_for_background_tasks(task_logger)

    # Verify the task was logged with errors
    async with session_maker() as db_session:
        from kavalai.agents.db import Task
        from sqlalchemy import select

        result = await db_session.execute(select(Task).where(Task.run_id == run.id))
        tasks = result.scalars().all()

        assert len(tasks) == 1
        task = tasks[0]
        assert task.name == "test_llm_task_error"
        assert task.errors == ["Validation error", "Timeout error"]


@pytest.mark.asyncio
async def test_task_logger_agent_task(agent_service, session_maker):
    """Test logging an agent task with system prompt."""
    # Create agent, session, and run
    agent = await agent_service.get_or_create_agent(
        name="test_agent",
        description="Test agent",
        input_schema={},
        output_schema={},
        workflow={},
    )
    session = await agent_service.get_or_create_session(agent_id=agent.id)
    run = await agent_service.create_run(session_id=session.id, input_data={})

    # Create run context
    run_context = RunContext(agent_service=agent_service)
    run_context.agent_id = agent.id
    run_context.session_id = session.id
    run_context.run_id = run.id

    # Create task logger and log an agent task
    task_logger = TaskLogger(agent_service, run_context)
    system_prompt = """You are a planning agent.
# Available tools:
- tool1
- tool2

# Inputs:
{"query": "test"}
"""
    task_logger.log_agent_task(
        task_name="planning_agent_task",
        system_prompt=system_prompt,
        input_data={"query": "test"},
        output={"result": "success"},
        duration=5.2,
    )
    await wait_for_background_tasks(task_logger)

    # Verify the task was logged with system prompt
    async with session_maker() as db_session:
        from kavalai.agents.db import Task
        from sqlalchemy import select

        result = await db_session.execute(select(Task).where(Task.run_id == run.id))
        tasks = result.scalars().all()

        assert len(tasks) == 1
        task = tasks[0]
        assert task.name == "planning_agent_task"
        assert task.prompt == system_prompt
        assert task.inputs == {"query": "test"}
        assert task.output == {"result": "success"}
        assert task.duration_seconds == 5.2


@pytest.mark.asyncio
async def test_task_logger_tool_call(agent_service, session_maker):
    """Test logging a tool call."""
    # Create agent, session, and run
    agent = await agent_service.get_or_create_agent(
        name="test_agent",
        description="Test agent",
        input_schema={},
        output_schema={},
        workflow={},
    )
    session = await agent_service.get_or_create_session(agent_id=agent.id)
    run = await agent_service.create_run(session_id=session.id, input_data={})

    # Create run context
    run_context = RunContext(agent_service=agent_service)
    run_context.agent_id = agent.id
    run_context.session_id = session.id
    run_context.run_id = run.id

    # Create task logger and log a tool call
    task_logger = TaskLogger(agent_service, run_context)
    task_logger.log_tool_call(
        tool_uri="python://websearch.serper_web_search",
        arguments={"query": "Kaval AI"},
        output={"results": ["result1", "result2"]},
        duration=2.1,
    )
    await wait_for_background_tasks(task_logger)

    # Verify the tool call was logged
    async with session_maker() as db_session:
        from kavalai.agents.db import Task
        from sqlalchemy import select

        result = await db_session.execute(select(Task).where(Task.run_id == run.id))
        tasks = result.scalars().all()

        assert len(tasks) == 1
        task = tasks[0]
        assert task.name == "python://websearch.serper_web_search"
        assert task.inputs == {"arguments": {"query": "Kaval AI"}}
        assert task.output == {"results": ["result1", "result2"]}
        assert task.duration_seconds == 2.1


@pytest.mark.asyncio
async def test_task_logger_rag_query(agent_service, session_maker):
    """Test logging a RAG query."""
    # Create agent, session, and run
    agent = await agent_service.get_or_create_agent(
        name="test_agent",
        description="Test agent",
        input_schema={},
        output_schema={},
        workflow={},
    )
    session = await agent_service.get_or_create_session(agent_id=agent.id)
    run = await agent_service.create_run(session_id=session.id, input_data={})

    # Create run context
    run_context = RunContext(agent_service=agent_service)
    run_context.agent_id = agent.id
    run_context.session_id = session.id
    run_context.run_id = run.id

    # Create task logger and log a RAG query
    task_logger = TaskLogger(agent_service, run_context)
    task_logger.log_rag_query(
        task_name="search_docs",
        query_text="What is Kaval AI?",
        top_k=5,
        collection_name="documentation",
        source_ids=["doc1", "doc2"],
        keep_best=True,
        output=[
            {"similarity": 0.95, "content": "Kaval AI is...", "source_id": "doc1"},
            {"similarity": 0.89, "content": "Kaval provides...", "source_id": "doc2"},
        ],
        duration=0.3,
    )
    await wait_for_background_tasks(task_logger)

    # Verify the RAG query was logged
    async with session_maker() as db_session:
        from kavalai.agents.db import Task
        from sqlalchemy import select

        result = await db_session.execute(select(Task).where(Task.run_id == run.id))
        tasks = result.scalars().all()

        assert len(tasks) == 1
        task = tasks[0]
        assert task.name == "search_docs"
        assert task.inputs == {
            "text": "What is Kaval AI?",
            "top_k": 5,
            "collection_name": "documentation",
            "source_ids": ["doc1", "doc2"],
            "keep_best": True,
        }
        assert len(task.output) == 2
        assert task.output[0]["similarity"] == 0.95
        assert task.duration_seconds == 0.3


@pytest.mark.asyncio
async def test_task_logger_without_agent_service():
    """Test that TaskLogger gracefully handles None agent_service."""
    # Create run context without agent service
    run_context = RunContext(agent_service=None)

    # Create task logger with None agent_service
    task_logger = TaskLogger(None, run_context)

    # Should not raise an error, just silently skip logging
    task_logger.log_llm_task(
        task_name="test_task",
        prompt="test",
        input_data={},
        output={},
        duration=1.0,
    )

    task_logger.log_tool_call(
        tool_uri="test://tool",
        arguments={},
        output={},
        duration=1.0,
    )
    await wait_for_background_tasks(task_logger)


@pytest.mark.asyncio
async def test_task_logger_tool_call_with_errors(agent_service, session_maker):
    """Test logging a tool call with errors."""
    # Create agent, session, and run
    agent = await agent_service.get_or_create_agent(
        name="test_agent",
        description="Test agent",
        input_schema={},
        output_schema={},
        workflow={},
    )
    session = await agent_service.get_or_create_session(agent_id=agent.id)
    run = await agent_service.create_run(session_id=session.id, input_data={})

    # Create run context
    run_context = RunContext(agent_service=agent_service)
    run_context.agent_id = agent.id
    run_context.session_id = session.id
    run_context.run_id = run.id

    # Create task logger and log a tool call with errors
    task_logger = TaskLogger(agent_service, run_context)
    task_logger.log_tool_call(
        tool_uri="python://failing_tool",
        arguments={"param": "value"},
        output="Error: Tool execution failed",
        duration=0.5,
        errors=["ValidationError: Missing required field", "Timeout after 30s"],
    )
    await wait_for_background_tasks(task_logger)

    # Verify the tool call was logged with errors
    async with session_maker() as db_session:
        from kavalai.agents.db import Task
        from sqlalchemy import select

        result = await db_session.execute(select(Task).where(Task.run_id == run.id))
        tasks = result.scalars().all()

        assert len(tasks) == 1
        task = tasks[0]
        assert task.name == "python://failing_tool"
        assert task.errors == [
            "ValidationError: Missing required field",
            "Timeout after 30s",
        ]
        assert "Error:" in str(task.output)


@pytest.mark.asyncio
async def test_task_logger_handles_logging_exceptions(
    agent_service, session_maker, caplog
):
    """Test that TaskLogger handles exceptions during logging gracefully."""
    import logging
    from unittest.mock import AsyncMock

    # Create agent, session, and run
    agent = await agent_service.get_or_create_agent(
        name="test_agent",
        description="Test agent",
        input_schema={},
        output_schema={},
        workflow={},
    )
    session = await agent_service.get_or_create_session(agent_id=agent.id)
    run = await agent_service.create_run(session_id=session.id, input_data={})

    # Create run context
    run_context = RunContext(agent_service=agent_service)
    run_context.agent_id = agent.id
    run_context.session_id = session.id
    run_context.run_id = run.id

    # Mock add_task to raise an exception
    original_add_task = agent_service.add_task
    agent_service.add_task = AsyncMock(
        side_effect=Exception("Database connection failed")
    )

    # Create task logger
    task_logger = TaskLogger(agent_service, run_context)

    # Should not raise an exception, but log the error
    with caplog.at_level(logging.ERROR):
        task_logger.log_tool_call(
            tool_uri="python://test_tool",
            arguments={},
            output={},
            duration=1.0,
        )
        await wait_for_background_tasks(task_logger)

    # Verify error was logged
    assert any(
        "Background task logging failed" in record.message
        for record in caplog.records
    )

    # Restore original method
    agent_service.add_task = original_add_task
