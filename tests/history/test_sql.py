"""Tests for the built-in SQL HistoryService backend."""

import pytest
from uuid import uuid4

from kavalai.agents.agent_service import AgentService
from kavalai.agents.db import Agent, ModelCallStat, Session
from kavalai.history import HistoryService, SqlHistoryService


async def make_session(agents_db) -> tuple:
    """Create an agent + session (core rows history records point at)."""
    agent = Agent(name=f"history-test-{uuid4()}")
    agents_db.add(agent)
    await agents_db.commit()
    session = Session(agent_id=agent.id)
    agents_db.add(session)
    await agents_db.commit()
    return agent, session


@pytest.mark.asyncio
async def test_chat_history_roundtrip(agents_db, agents_session_maker):
    service = SqlHistoryService(agents_session_maker)
    agent, session = await make_session(agents_db)

    for i in range(5):
        await service.add_chat_message(
            agent_id=agent.id,
            session_id=session.id,
            role="user" if i % 2 == 0 else "assistant",
            content=f"message {i}",
        )

    messages = await service.get_chat_messages(session.id)
    assert [m.content for m in messages] == [f"message {i}" for i in range(5)]
    assert messages[0].role == "user"

    # Windowing
    window = await service.get_chat_messages(session.id, limit=2)
    assert len(window) == 2


@pytest.mark.asyncio
async def test_task_and_stats(agents_db, agents_session_maker):
    service = SqlHistoryService(agents_session_maker)
    agent, session = await make_session(agents_db)

    from kavalai.agents.agent_service import AgentService as _AS  # noqa: F401

    # A run row for the task to reference
    agent_service = AgentService(agents_session_maker)
    run = await agent_service.create_run(session_id=session.id, input_data={})

    task = await service.add_task(
        session_id=session.id,
        run_id=run.id,
        agent_id=agent.id,
        name="node-1",
        node_type="prompt",
        inputs={"q": 1},
        output={"a": 2},
        duration_seconds=0.5,
    )
    assert task.id is not None
    assert task.name == "node-1"

    stat = await service.add_model_call_stat(
        ModelCallStat(call_type="llm", model="test/model"), agent_id=agent.id
    )
    assert stat.agent_id == agent.id

    stats = await service.get_model_call_stats(call_type="llm", limit=10)
    assert any(s.id == stat.id for s in stats)
    assert await service.get_model_call_stats(call_type="does-not-exist") == []


@pytest.mark.asyncio
async def test_deletion_semantics(agents_db, agents_session_maker):
    service = SqlHistoryService(agents_session_maker)
    agent, session = await make_session(agents_db)
    _, other_session = await make_session(agents_db)

    for sess in (session, other_session):
        await service.add_chat_message(
            agent_id=agent.id if sess is session else sess.agent_id,
            session_id=sess.id,
            role="user",
            content="hello",
        )

    await service.delete_for_session(session.id)
    assert await service.get_chat_messages(session.id) == []
    assert len(await service.get_chat_messages(other_session.id)) == 1

    # delete_for_agent removes stats too
    await service.add_model_call_stat(
        ModelCallStat(call_type="llm", model="m"), agent_id=agent.id
    )
    await service.delete_for_agent(agent.id)
    stats = await service.get_model_call_stats(call_type="llm", limit=100)
    assert not any(s.agent_id == agent.id for s in stats)


@pytest.mark.asyncio
async def test_agent_service_delegates_to_history(agents_db, agents_session_maker):
    """AgentService is a facade: history ops flow through HistoryService."""
    agent_service = AgentService(agents_session_maker)
    assert isinstance(agent_service.history, HistoryService)
    assert isinstance(agent_service.history, SqlHistoryService)

    agent, session = await make_session(agents_db)
    message = await agent_service.add_chat_message(
        agent_id=agent.id, session_id=session.id, role="user", content="hi"
    )
    history = await agent_service.get_chat_history(session.id)
    assert [m.id for m in history] == [message.id]

    # A custom backend can be injected.
    class Recorder(SqlHistoryService):
        def __init__(self, inner):
            super().__init__(inner.session_maker)
            self.calls = 0

        async def add_chat_message(self, *args, **kwargs):
            self.calls += 1
            return await super().add_chat_message(*args, **kwargs)

    recorder = Recorder(agent_service.history)
    custom = AgentService(agents_session_maker, history=recorder)
    await custom.add_chat_message(
        agent_id=agent.id, session_id=session.id, role="user", content="via custom"
    )
    assert recorder.calls == 1
