import pytest
from kavalai.agents.agent_service import AgentService
from kavalai.agents.task_logger import TaskLogger
from kavalai.agents.run_context import RunContext


@pytest.mark.asyncio
async def test_reproduce_null_char_issue(agents_session_maker):
    """
    Test that strings containing null characters (\u0000) can be logged
    without causing a database error.
    """
    service = AgentService(agents_session_maker)
    agent = await service.get_or_create_agent(name="NullCharAgent")
    session = await service.get_or_create_session(agent_id=agent.id)
    run = await service.create_run(session_id=session.id)

    run_context = RunContext(
        agent_id=agent.id, session_id=session.id, run_id=run.id, data={}
    )

    task_logger = TaskLogger(agent_service=service, run_context=run_context)

    # This string contains a null character, which caused the reported error
    problematic_string = "Analyzing with a null character: \u0000 and some more text."

    # Try to log an agent task with the problematic string in various fields,
    # including a null character in a dictionary key.
    await task_logger.log_agent_task(
        task_name="Analyzing\u0000Task",
        system_prompt=problematic_string,
        input_data={
            "user_input": "Input with null\u0000",
            "key\u0000with\u0000null": "value",
        },
        output={
            "agent_response": "Output with null\u0000",
            "nested": {"another\u0000key": "val"},
        },
        duration=1.0,
        errors=["Error with null\u0000"],
    )

    # If we reach here without an exception, the immediate crash is avoided.
    # Now check if the data was actually saved (and cleaned).
    async with agents_session_maker() as db_session:
        from kavalai.agents.db import Task
        from sqlalchemy import select

        stmt = select(Task).where(Task.run_id == run.id)
        result = await db_session.execute(stmt)
        task = result.scalars().first()

        assert task is not None
        assert "\u0000" not in task.name
        assert "\u0000" not in task.prompt
        assert "\u0000" not in str(task.inputs)
        assert "\u0000" not in str(task.output)
        assert "\u0000" not in str(task.errors)

        # Verify they were cleaned (removed)
        assert "AnalyzingTask" == task.name
        assert "Analyzing with a null character:  and some more text." == task.prompt
        assert "Input with null" in str(task.inputs)
        assert "Output with null" in str(task.output)
        assert "Error with null" in str(task.errors)

        # Verify keys were cleaned
        assert "keywithnull" in task.inputs
        assert "anotherkey" in task.output["nested"]


@pytest.mark.asyncio
async def test_reproduce_null_char_in_model_call_stats(agents_session_maker):
    from kavalai.llm_clients.common import create_model_call_stat
    from kavalai.agents.agent_service import AgentService

    service = AgentService(agents_session_maker)
    agent = await service.get_or_create_agent(name="NullCharAgentStats")

    stats = create_model_call_stat(
        call_type="llm",
        model="openai/gpt-4",
        duration_sections=1.0,
        response_data={"key\u0000": "value\u0000"},
    )

    # This should not crash
    await service.add_model_call_stats(stats=stats, agent_id=agent.id)

    # Verify it was cleaned
    async with agents_session_maker() as session:
        from kavalai.agents.db import ModelCallStat
        from sqlalchemy import select

        stmt = select(ModelCallStat).where(ModelCallStat.agent_id == agent.id)
        result = await session.execute(stmt)
        saved_stats = result.scalars().first()

        assert saved_stats is not None
        assert saved_stats.response_data == {"key": "value"}


@pytest.mark.asyncio
async def test_reproduce_null_char_in_chat_message(agents_session_maker):
    service = AgentService(agents_session_maker)
    agent = await service.get_or_create_agent(name="NullCharAgentChat")
    session = await service.get_or_create_session(agent_id=agent.id)

    # Add chat message with null char
    await service.add_chat_message(
        agent_id=agent.id,
        session_id=session.id,
        role="user",
        content="Hello\u0000World",
    )

    history = await service.get_chat_history(session.id)
    assert len(history) == 1
    assert history[0].content == "HelloWorld"
    assert "\u0000" not in history[0].content
