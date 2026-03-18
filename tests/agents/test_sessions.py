import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from kavalai.agents.db import Agent, Session, Run, Task, ChatMessage
from kavalai.agents.sessions import get_sessions_summary, get_session_details


@pytest.mark.asyncio
async def test_get_session_details(agents_db):
    agent = Agent(id=uuid4(), name="Test Agent")
    agents_db.add(agent)
    await agents_db.commit()

    now = datetime.now(timezone.utc)
    s1 = Session(id=uuid4(), agent_id=agent.id, created_at=now, updated_at=now)

    r1 = Run(
        id=uuid4(),
        session_id=s1.id,
        input_data={"q": "test"},
        output_data={"ans": "res"},
        created_at=now - timedelta(minutes=5),
    )
    m1 = ChatMessage(
        id=uuid4(),
        agent_id=agent.id,
        session_id=s1.id,
        run_id=r1.id,
        role="user",
        content="Hello",
        created_at=now - timedelta(minutes=4),
    )
    t1 = Task(
        id=uuid4(),
        agent_id=agent.id,
        session_id=s1.id,
        run_id=r1.id,
        inputs={"cmd": "ls"},
        output={"out": "file1"},
        name="test_task",
        created_at=now - timedelta(minutes=3),
    )

    agents_db.add_all([s1, r1, m1, t1])
    await agents_db.commit()

    details = await get_session_details(agents_db, s1.id)

    assert details.session_id == s1.id
    assert len(details.messages) == 1
    assert details.messages[0].content == "Hello"
    assert len(details.runs) == 1
    assert details.runs[0].id == r1.id
    assert details.runs[0].tasks_count == 1
    assert len(details.tasks) == 1
    assert details.tasks[0].id == t1.id
    assert details.tasks[0].name == "test_task"
    assert details.tasks[0].run_id == r1.id


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
    result = await get_sessions_summary(agents_db)
    summaries = result["sessions"]
    total_count = result["total_count"]

    # 3. Assertions
    assert len(summaries) == 2
    assert total_count == 2

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

    result_filtered = await get_sessions_summary(agents_db, agent_id=another_agent.id)
    summaries_filtered = result_filtered["sessions"]
    assert len(summaries_filtered) == 1
    assert result_filtered["total_count"] == 1
    assert summaries_filtered[0].session_id == s3.id


@pytest.mark.asyncio
async def test_get_sessions_summary_with_search(agents_db):
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
        content="Looking for a specific needle in the haystack",
        created_at=now,
    )

    s2 = Session(id=uuid4(), agent_id=agent.id, created_at=now, updated_at=now)
    m2 = ChatMessage(
        id=uuid4(),
        agent_id=agent.id,
        session_id=s2.id,
        role="user",
        content="Just a normal message",
        created_at=now,
    )

    agents_db.add_all([s1, m1, s2, m2])
    await agents_db.commit()

    # Search for "needle"
    result = await get_sessions_summary(agents_db, search="needle")
    assert result["total_count"] == 1
    assert result["sessions"][0].session_id == s1.id

    # Search for "MESSAGE" (case insensitive)
    result = await get_sessions_summary(agents_db, search="MESSAGE")
    assert result["total_count"] == 1
    assert result["sessions"][0].session_id == s2.id

    # Search for something non-existent
    result = await get_sessions_summary(agents_db, search="nonexistent")
    assert result["total_count"] == 0
    assert len(result["sessions"]) == 0


@pytest.mark.asyncio
async def test_get_sessions_summary_with_date_range(agents_db):
    agent = Agent(id=uuid4(), name="Test Agent")
    agents_db.add(agent)
    await agents_db.commit()

    now = datetime.now(timezone.utc).replace(microsecond=0)

    # Session 1: Today
    s1 = Session(id=uuid4(), agent_id=agent.id, created_at=now, updated_at=now)

    # Session 2: 10 days ago
    s2 = Session(
        id=uuid4(),
        agent_id=agent.id,
        created_at=now - timedelta(days=10),
        updated_at=now - timedelta(days=10),
    )

    agents_db.add_all([s1, s2])
    await agents_db.commit()

    # Filter for last 7 days
    start_date = now - timedelta(days=7)
    result = await get_sessions_summary(agents_db, start_date=start_date)
    assert result["total_count"] == 1
    assert result["sessions"][0].session_id == s1.id

    # Filter for range that includes only s2
    end_date = now - timedelta(days=5)
    result = await get_sessions_summary(agents_db, end_date=end_date)
    assert result["total_count"] == 1
    assert result["sessions"][0].session_id == s2.id

    # Filter for both
    result = await get_sessions_summary(
        agents_db, start_date=now - timedelta(days=15), end_date=now + timedelta(days=1)
    )
    assert result["total_count"] == 2
