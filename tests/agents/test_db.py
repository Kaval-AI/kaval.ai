import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from kavalai.agents.db import (
    Agent,
    Session,
    Run,
    Task,
    ChatMessage,
    LLMProfile,
    LLMCallStat,
    ensure_async_scheme,
)
from kavalai.crud import insert, delete, get_one


@pytest.mark.asyncio
async def test_session_run_task_flow(agents_db: AsyncSession):
    """Test the hierarchy: Agent -> Session -> Run -> Task."""
    # 1. Setup Agent
    agent = await insert(agents_db, Agent, {"name": "Bot"})

    # 2. Create Session
    session = await insert(
        agents_db, Session, {"agent_id": agent.id, "external_id": "user_123"}
    )
    assert session.agent_id == agent.id

    # 3. Create Run (Execution)
    run = await insert(
        agents_db,
        Run,
        {
            "session_id": session.id,
            "input_data": {"query": "Execute task"},
            "context": {"metadata": "test-context"},
        },
    )
    assert run.session_id == session.id

    # 4. Create Task linked to the run
    task = await insert(
        agents_db,
        Task,
        {
            "agent_id": agent.id,
            "session_id": session.id,
            "run_id": run.id,
            "inputs": {"sub_task": "process_data"},
        },
    )
    assert task.run_id == run.id

    # 5. Create Chat Message linked to run
    message = await insert(
        agents_db,
        ChatMessage,
        {
            "agent_id": agent.id,
            "session_id": session.id,
            "run_id": run.id,
            "role": "assistant",
            "content": "Hello there!",
        },
    )
    assert message.run_id == run.id


@pytest.mark.asyncio
async def test_run_set_null_on_delete_for_messages(agents_db: AsyncSession):
    """Ensure deleting a run sets run_id to NULL in messages instead of deleting them."""
    agent = await insert(agents_db, Agent, {"name": "Persistence Test"})
    session = await insert(agents_db, Session, {"agent_id": agent.id})
    run = await insert(agents_db, Run, {"session_id": session.id})

    message = await insert(
        agents_db,
        ChatMessage,
        {
            "agent_id": agent.id,
            "session_id": session.id,
            "run_id": run.id,
            "role": "assistant",
            "content": "I am linked to a run",
        },
    )

    # FIX: Capture the ID before expiring the object
    message_id = message.id

    # Action: Delete ONLY the run
    await delete(agents_db, Run, run.id)

    # This detaches the local 'message' object from its current state
    agents_db.expire_all()

    # Asserts - Use the local message_id variable
    fetched_msg = await get_one(agents_db, ChatMessage, message_id)

    assert fetched_msg is not None
    assert fetched_msg.run_id is None  # Check ON DELETE SET NULL works


@pytest.mark.asyncio
async def test_llm_profile_and_stats(agents_db: AsyncSession):
    """Test LLMProfile and LLMCallStat models and their relationship."""
    # 1. Create LLM Profile
    profile = await insert(
        agents_db,
        LLMProfile,
        {
            "name": "Test OpenAI",
            "provider": "openai",
            "model_name": "gpt-4",
            "api_key": "sk-test",
            "config": {"mode": "TOOLS"},
        },
    )
    assert profile.name == "Test OpenAI"

    # 2. Create Call Stat linked to profile
    stat = await insert(
        agents_db,
        LLMCallStat,
        {
            "llm_profile_id": profile.id,
            "response_code": 200,
            "cost": 0.000123,
        },
    )
    assert stat.llm_profile_id == profile.id

    # 3. Test Relationship (LLMProfile -> LLMCallStat)
    # Re-fetch profile to load relationship
    await get_one(agents_db, LLMProfile, profile.id)

    # 4. Test ON DELETE SET NULL
    profile_id = profile.id
    stat_id = stat.id

    # Action: Delete ONLY the profile
    await delete(agents_db, LLMProfile, profile_id)

    # Clear session to force re-fetch from DB
    agents_db.expire_all()

    assert await get_one(agents_db, LLMProfile, profile_id) is None

    # Re-fetch stat to see if it's still there but with llm_profile_id=None
    fetched_stat = await get_one(agents_db, LLMCallStat, stat_id)
    assert fetched_stat is not None
    assert fetched_stat.llm_profile_id is None


def test_ensure_async_scheme():
    """Test the ensure_async_scheme utility function."""
    # Test converting standard postgresql scheme
    assert (
        ensure_async_scheme("postgresql://user:pass@host:5432/db")
        == "postgresql+asyncpg://user:pass@host:5432/db"
    )

    # Test with already correct scheme
    assert (
        ensure_async_scheme("postgresql+asyncpg://user:pass@host:5432/db")
        == "postgresql+asyncpg://user:pass@host:5432/db"
    )

    # Test with other postgresql variations
    assert (
        ensure_async_scheme("postgresql+psycopg2://user:pass@host:5432/db")
        == "postgresql+asyncpg://user:pass@host:5432/db"
    )

    # Test with non-postgresql scheme
    assert ensure_async_scheme("sqlite:///:memory:") == "sqlite:///:memory:"

    # Test with invalid URI
    assert ensure_async_scheme("not_a_uri") == "not_a_uri"
    assert ensure_async_scheme("") == ""
    assert ensure_async_scheme(None) is None
