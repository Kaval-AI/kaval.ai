import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from kavalai.agents.db import Agent, Session, Run, Task, ChatMessage
from kavalai.agents.sessions import get_sessions_summary, get_session_messages


@pytest.mark.asyncio
async def test_get_session_messages(agents_db):
    agent = Agent(id=uuid4(), name="Test Agent")
    agents_db.add(agent)
    await agents_db.commit()

    now = datetime.now(timezone.utc)
    s1 = Session(id=uuid4(), agent_id=agent.id, created_at=now, updated_at=now)

    m1 = ChatMessage(
        id=uuid4(),
        agent_id=agent.id,
        session_id=s1.id,
        role="user",
        content="Hello",
        created_at=now - timedelta(minutes=2),
    )
    m2 = ChatMessage(
        id=uuid4(),
        agent_id=agent.id,
        session_id=s1.id,
        role="assistant",
        content="Hi there!",
        created_at=now - timedelta(minutes=1),
    )

    agents_db.add_all([s1, m1, m2])
    await agents_db.commit()

    messages = await get_session_messages(agents_db, s1.id)

    assert len(messages) == 2
    assert messages[0].content == "Hello"
    assert messages[1].content == "Hi there!"
    assert messages[0].created_at < messages[1].created_at


@pytest.mark.asyncio
async def test_get_sessions_summary(agents_db):
    # 1. Setup data
    agent = Agent(id=uuid4(), name="Test Agent")
    agents_db.add(agent)
    await agents_db.commit()

    now = datetime.now(timezone.utc)

    # Session 1: 1 run, 1 task, 2 messages
    s1 = Session(id=uuid4(), agent_id=agent.id, created_at=now, updated_at=now)
    r1 = Run(id=uuid4(), session_id=s1.id, created_at=now)
    t1 = Task(
        id=uuid4(), agent_id=agent.id, session_id=s1.id, run_id=r1.id, created_at=now
    )
    m1 = ChatMessage(
        id=uuid4(),
        agent_id=agent.id,
        session_id=s1.id,
        run_id=r1.id,
        role="user",
        content="First message",
        created_at=now - timedelta(minutes=5),
    )
    m2 = ChatMessage(
        id=uuid4(),
        agent_id=agent.id,
        session_id=s1.id,
        run_id=r1.id,
        role="assistant",
        content="Last message",
        created_at=now,
    )

    # Session 2: No runs, no tasks, no messages
    s2 = Session(
        id=uuid4(),
        agent_id=agent.id,
        created_at=now - timedelta(hours=1),
        updated_at=now - timedelta(hours=1),
    )

    agents_db.add_all([s1, r1, t1, m1, m2, s2])
    await agents_db.commit()

    # 2. Call the function
    summaries = await get_sessions_summary(agents_db)

    # 3. Assertions
    assert len(summaries) == 2

    # Ordered by updated_at desc, so s1 should be first
    summary1 = summaries[0]
    assert summary1.session_id == s1.id
    assert summary1.agent_name == "Test Agent"
    assert summary1.runs_count == 1
    assert summary1.tasks_count == 1
    assert summary1.messages_count == 2
    assert summary1.first_message == "First message"
    assert summary1.last_message == "Last message"

    summary2 = summaries[1]
    assert summary2.session_id == s2.id
    assert summary2.runs_count == 0
    assert summary2.tasks_count == 0
    assert summary2.messages_count == 0
    assert summary2.first_message is None
    assert summary2.last_message is None

    # Test filtering by agent_id
    another_agent = Agent(id=uuid4(), name="Another Agent")
    agents_db.add(another_agent)
    await agents_db.commit()

    s3 = Session(id=uuid4(), agent_id=another_agent.id, created_at=now, updated_at=now)
    agents_db.add(s3)
    await agents_db.commit()

    summaries_filtered = await get_sessions_summary(
        agents_db, agent_id=another_agent.id
    )
    assert len(summaries_filtered) == 1
    assert summaries_filtered[0].session_id == s3.id
