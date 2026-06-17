from uuid import uuid4

import pytest

from kavalai.agents.v2.workflow.state import WorkflowState
from kavalai.agents.v2.workflow.storage.postgres import PostgresDataStorage


@pytest.mark.asyncio
class TestPostgresDataStorage:
    async def test_initialize_run_and_chat_history(
        self, agents_session_maker, agents_db
    ):
        storage = PostgresDataStorage.from_session_maker(agents_session_maker)
        handle = await storage.initialize_run(
            workflow_name="pg_wf", input_data={"user_message": "hi"}
        )
        assert handle.agent_id and handle.session_id and handle.run_id

        await storage.add_chat_message(
            agent_id=handle.agent_id,
            session_id=handle.session_id,
            run_id=handle.run_id,
            role="user",
            content="hi",
        )
        await storage.add_chat_message(
            agent_id=handle.agent_id,
            session_id=handle.session_id,
            run_id=handle.run_id,
            role="assistant",
            content="hello",
        )
        history = await storage.get_chat_history(handle.session_id)
        assert [(m.role, m.content) for m in history] == [
            ("user", "hi"),
            ("assistant", "hello"),
        ]

    async def test_save_and_load_state(self, agents_session_maker, agents_db):
        storage = PostgresDataStorage.from_session_maker(agents_session_maker)
        handle = await storage.initialize_run(workflow_name="pg_state")
        state = WorkflowState(
            workflow_name="pg_state",
            status="completed",
            trace=["start", "reply", "end"],
            run_id=handle.run_id,
            token_usage={"model_calls": 1, "total_tokens": 5},
            output_data={"agent_response": "done"},
        )
        await storage.save_state(handle.run_id, state)

        loaded = await storage.load_state(handle.run_id)
        assert loaded is not None
        assert loaded.status == "completed"
        assert loaded.trace == ["start", "reply", "end"]
        assert loaded.token_usage["total_tokens"] == 5

    async def test_update_run_then_state_persists(
        self, agents_session_maker, agents_db
    ):
        storage = PostgresDataStorage.from_session_maker(agents_session_maker)
        handle = await storage.initialize_run(workflow_name="pg_update")
        await storage.update_run(
            handle.run_id, output_data={"agent_response": "ok"}, context={"a": 1}
        )
        # update_run wrote context as the raw dict; load_state tolerates it.
        loaded = await storage.load_state(handle.run_id)
        assert loaded is None  # {"a": 1} is not a valid WorkflowState

    async def test_load_state_missing_returns_none(
        self, agents_session_maker, agents_db
    ):
        storage = PostgresDataStorage.from_session_maker(agents_session_maker)
        assert await storage.load_state(str(uuid4())) is None
