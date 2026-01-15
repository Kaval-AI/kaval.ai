import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from kavalai.agents.db import Agent, Session, Run, ChatMessage
from kavalai.agents.stats import get_daily_stats


@pytest.mark.asyncio
async def test_get_daily_stats(agents_db):
    # 1. Setup data
    agent = Agent(id=uuid4(), name="Test Agent")
    agents_db.add(agent)
    await agents_db.commit()

    now = datetime.now(timezone.utc)

    # Create some sessions, runs, and messages on different days
    # Today
    s1 = Session(id=uuid4(), agent_id=agent.id, created_at=now)
    r1 = Run(id=uuid4(), session_id=s1.id, created_at=now)
    m1 = ChatMessage(
        id=uuid4(),
        agent_id=agent.id,
        session_id=s1.id,
        role="user",
        content="hi",
        created_at=now,
    )

    # Yesterday
    yesterday = now - timedelta(days=1)
    s2 = Session(id=uuid4(), agent_id=agent.id, created_at=yesterday)
    r2 = Run(id=uuid4(), session_id=s2.id, created_at=yesterday)
    m2 = ChatMessage(
        id=uuid4(),
        agent_id=agent.id,
        session_id=s2.id,
        role="user",
        content="yesterday",
        created_at=yesterday,
    )

    # 8 days ago (should be excluded)
    eight_days_ago = now - timedelta(days=8)
    s3 = Session(id=uuid4(), agent_id=agent.id, created_at=eight_days_ago)

    agents_db.add_all([s1, r1, m1, s2, r2, m2, s3])
    await agents_db.commit()

    # 2. Call the function
    stats = await get_daily_stats(agents_db, days=7)

    # 3. Assertions
    assert "runs" in stats
    assert "sessions" in stats
    assert "messages" in stats

    assert len(stats["runs"]) == 7
    assert len(stats["sessions"]) == 7
    assert len(stats["messages"]) == 7

    # Check today's counts (the last element in the series)
    assert stats["sessions"][-1]["count"] == 1
    assert stats["runs"][-1]["count"] == 1
    assert stats["messages"][-1]["count"] == 1

    # Check yesterday's counts (the second to last element)
    assert stats["sessions"][-2]["count"] == 1
    assert stats["runs"][-2]["count"] == 1
    assert stats["messages"][-2]["count"] == 1

    # Check that 8 days ago is NOT counted (all other days should be 0)
    for i in range(5):  # first 5 days of the 7-day period
        assert stats["sessions"][i]["count"] == 0
