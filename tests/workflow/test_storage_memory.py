import pytest

from kavalai.workflow.state import WorkflowState
from kavalai.workflow.storage.memory import SqliteDataStorage


@pytest.fixture
async def storage():
    s = SqliteDataStorage()
    yield s
    await s.close()


async def test_initialize_run_creates_ids(storage):
    handle = await storage.initialize_run(
        workflow_name="wf", input_data={"user_message": "hi"}
    )
    assert handle.agent_id
    assert handle.session_id
    assert handle.run_id


async def test_agent_is_reused_by_name(storage):
    h1 = await storage.initialize_run(workflow_name="wf")
    h2 = await storage.initialize_run(workflow_name="wf")
    assert h1.agent_id == h2.agent_id
    # New run + session each time though.
    assert h1.run_id != h2.run_id
    assert h1.session_id != h2.session_id


async def test_session_is_reused_when_supplied(storage):
    h1 = await storage.initialize_run(workflow_name="wf")
    h2 = await storage.initialize_run(workflow_name="wf", session_id=h1.session_id)
    assert h2.session_id == h1.session_id


async def test_supplied_unknown_session_id_is_kept(storage):
    handle = await storage.initialize_run(
        workflow_name="wf", session_id="custom-session"
    )
    assert handle.session_id == "custom-session"


async def test_save_and_load_state(storage):
    handle = await storage.initialize_run(workflow_name="wf")
    state = WorkflowState(
        workflow_name="wf", status="running", trace=["start"], run_id=handle.run_id
    )
    await storage.save_state(handle.run_id, state)
    loaded = await storage.load_state(handle.run_id)
    assert loaded == state


async def test_load_state_none_when_absent(storage):
    handle = await storage.initialize_run(workflow_name="wf")
    assert await storage.load_state(handle.run_id) is None
    assert await storage.load_state("does-not-exist") is None


async def test_update_run(storage):
    handle = await storage.initialize_run(workflow_name="wf")
    await storage.update_run(
        handle.run_id, output_data={"agent_response": "ok"}, context={"a": 1}
    )
    conn = await storage._connect()
    async with conn.execute(
        "SELECT output_data, context FROM runs WHERE id = ?", (handle.run_id,)
    ) as cur:
        row = await cur.fetchone()
    assert '"agent_response"' in row["output_data"]
    assert '"a"' in row["context"]


async def test_update_run_no_fields_is_noop(storage):
    handle = await storage.initialize_run(workflow_name="wf")
    await storage.update_run(handle.run_id)  # nothing to update
    conn = await storage._connect()
    async with conn.execute(
        "SELECT output_data, context FROM runs WHERE id = ?", (handle.run_id,)
    ) as cur:
        row = await cur.fetchone()
    assert row["output_data"] is None and row["context"] is None


async def test_chat_history_order(storage):
    handle = await storage.initialize_run(workflow_name="wf")
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


async def test_chat_history_limit(storage):
    handle = await storage.initialize_run(workflow_name="wf")
    for i in range(5):
        await storage.add_chat_message(
            agent_id=handle.agent_id,
            session_id=handle.session_id,
            run_id=handle.run_id,
            role="user",
            content=f"m{i}",
        )
    history = await storage.get_chat_history(handle.session_id, limit=2)
    assert [m.content for m in history] == ["m0", "m1"]


async def test_close_is_idempotent(storage):
    await storage.initialize_run(workflow_name="wf")
    await storage.close()
    await storage.close()  # no error on second close


async def test_base_close_default_returns_none():
    # Exercise the DataStorage ABC's default close() (no resources to release).
    from kavalai.workflow.storage.base import DataStorage, RunHandle

    class MinimalStorage(DataStorage):
        async def initialize_run(self, **kwargs):
            return RunHandle(agent_id="a", session_id="s", run_id="r")

        async def update_run(self, run_id, **kwargs):
            return None

        async def save_state(self, run_id, state):
            return None

        async def load_state(self, run_id):
            return None

        async def add_chat_message(self, **kwargs):
            return None

        async def get_chat_history(self, session_id, limit=50):
            return []

    assert await MinimalStorage().close() is None
