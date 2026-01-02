import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from kavalai.agents.db import Agent, Session, Interaction, ChatMessage
from kavalai.crud import insert, delete, get_one


@pytest.mark.asyncio
async def test_session_and_interaction_flow(agents_db: AsyncSession):
    """Test the hierarchy: Agent -> Session -> Interaction."""
    # 1. Setup Agent
    agent = await insert(agents_db, Agent, {"name": "Bot"})

    # 2. Create Session
    session = await insert(
        agents_db, Session, {"agent_id": agent.id, "external_id": "user_123"}
    )
    assert session.agent_id == agent.id

    # 3. Create Interaction
    interaction = await insert(
        agents_db,
        Interaction,
        {
            "session_id": session.id,
            "input_data": {"query": "hello"},
            "steps": {"thought": "Greeting", "action": "reply"},
        },
    )
    assert interaction.session_id == session.id

    # 4. Create Chat Message linked to interaction
    message = await insert(
        agents_db,
        ChatMessage,
        {
            "agent_id": agent.id,
            "session_id": session.id,
            "interaction_id": interaction.id,
            "role": "assistant",
            "content": "Hello there!",
        },
    )
    assert message.interaction_id == interaction.id


@pytest.mark.asyncio
async def test_agents_cascade_delete(agents_db: AsyncSession):
    """Ensure deleting an agent cleans up sessions and messages."""
    agent = await insert(agents_db, Agent, {"name": "Disposable"})
    session = await insert(agents_db, Session, {"agent_id": agent.id})
    interaction = await insert(agents_db, Interaction, {"session_id": session.id})

    # 1. Capture the IDs as local variables before deletion
    session_id = session.id
    interaction_id = interaction.id

    message = await insert(
        agents_db,
        ChatMessage,
        {
            "agent_id": agent.id,
            "session_id": session.id,
            "interaction_id": interaction.id,
            "role": "user",
            "content": "cascade test",
        },
    )
    message_id = message.id

    # 2. Action: Delete the Agent
    await delete(agents_db, Agent, agent.id)

    # 3. Clear the session
    agents_db.expire_all()

    # 4. Asserts - Use the local ID variables, NOT session.id
    assert await get_one(agents_db, Session, session_id) is None
    assert await get_one(agents_db, Interaction, interaction_id) is None
    assert await get_one(agents_db, ChatMessage, message_id) is None


@pytest.mark.asyncio
async def test_interaction_set_null_on_delete(agents_db: AsyncSession):
    """Ensure deleting an interaction sets interaction_id to NULL in messages instead of deleting them."""
    agent = await insert(agents_db, Agent, {"name": "Persistence Test"})
    session = await insert(agents_db, Session, {"agent_id": agent.id})
    interaction = await insert(agents_db, Interaction, {"session_id": session.id})

    message = await insert(
        agents_db,
        ChatMessage,
        {
            "agent_id": agent.id,
            "session_id": session.id,
            "interaction_id": interaction.id,
            "role": "assistant",
            "content": "I am linked to an interaction",
        },
    )

    # Action: Delete ONLY the interaction
    await delete(agents_db, Interaction, interaction.id)

    # Asserts
    fetched_msg = await get_one(agents_db, ChatMessage, message.id)
    assert fetched_msg is not None
    assert fetched_msg.interaction_id is None  # Check ON DELETE SET NULL
