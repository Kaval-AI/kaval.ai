import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from kavalai.agents.db import Agent, Session, Run, Task, ChatMessage
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
async def test_agents_cascade_delete(agents_db: AsyncSession):
    """Ensure deleting an agent cleans up sessions, runs, tasks and messages."""
    agent = await insert(agents_db, Agent, {"name": "Disposable"})
    session = await insert(agents_db, Session, {"agent_id": agent.id})
    run = await insert(agents_db, Run, {"session_id": session.id})
    task = await insert(
        agents_db,
        Task,
        {"agent_id": agent.id, "session_id": session.id, "run_id": run.id},
    )

    # Capture IDs
    session_id = session.id
    run_id = run.id
    task_id = task.id

    message = await insert(
        agents_db,
        ChatMessage,
        {
            "agent_id": agent.id,
            "session_id": session.id,
            "run_id": run_id,
            "role": "user",
            "content": "cascade test",
        },
    )
    message_id = message.id

    # Action: Delete the Agent
    await delete(agents_db, Agent, agent.id)
    agents_db.expire_all()

    # Asserts - Everything should be gone due to CASCADE
    assert await get_one(agents_db, Session, session_id) is None
    assert await get_one(agents_db, Run, run_id) is None
    assert await get_one(agents_db, Task, task_id) is None
    assert await get_one(agents_db, ChatMessage, message_id) is None


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
