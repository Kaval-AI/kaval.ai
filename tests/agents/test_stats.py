import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from kavalai.agents.db import (
    Agent,
    Session,
    Run,
    ChatMessage,
    LLMCallStat,
    EmbeddingCallStat,
)
from kavalai.agents.stats import get_daily_stats, get_summary_stats


@pytest.mark.asyncio
async def test_get_summary_stats(agents_db):
    agent = Agent(id=uuid4(), name="Test Agent")
    agents_db.add(agent)
    await agents_db.commit()

    now = datetime.now(timezone.utc)

    # 10 days ago
    ten_days_ago = now - timedelta(days=10)
    s1 = Session(id=uuid4(), agent_id=agent.id, created_at=ten_days_ago)
    stat1 = LLMCallStat(
        id=uuid4(),
        agent_id=agent.id,
        cost=0.05,
        created_at=ten_days_ago,
    )
    estat1 = EmbeddingCallStat(
        id=uuid4(),
        agent_id=agent.id,
        cost=0.01,
        created_at=ten_days_ago,
    )

    # 40 days ago (should be excluded)
    forty_days_ago = now - timedelta(days=40)
    s2 = Session(id=uuid4(), agent_id=agent.id, created_at=forty_days_ago)
    stat2 = LLMCallStat(
        id=uuid4(),
        agent_id=agent.id,
        cost=1.00,
        created_at=forty_days_ago,
    )

    agents_db.add_all([s1, stat1, estat1, s2, stat2])
    await agents_db.commit()

    stats = await get_summary_stats(agents_db)

    assert pytest.approx(stats["total_cost"]) == 0.06
    assert pytest.approx(stats["llm_cost"]) == 0.05
    assert pytest.approx(stats["embedding_cost"]) == 0.01
    assert stats["total_sessions"] == 1

    # Test with agent_id filter
    stats_agent = await get_summary_stats(agents_db, agent_id=agent.id)
    assert pytest.approx(stats_agent["total_cost"]) == 0.06
    assert pytest.approx(stats_agent["llm_cost"]) == 0.05
    assert pytest.approx(stats_agent["embedding_cost"]) == 0.01
    assert stats_agent["total_sessions"] == 1

    # Test with non-existent agent_id
    stats_none = await get_summary_stats(agents_db, agent_id=uuid4())
    assert stats_none["total_cost"] == 0.0
    assert stats_none["total_sessions"] == 0


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
