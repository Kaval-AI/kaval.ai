import pytest
from sqlalchemy import select

from kavalai.agents.db import ModelCallStat, Task
from kavalai.agents.v2.workflow.storage.postgres import PostgresDataStorage
from kavalai.agents.v2.workflow.tasklog.postgres import PostgresTaskLogger, _to_orm_stat
from kavalai.llm_clients.base_client import ModelCallStat as PydModelCallStat


def test_to_orm_stat_converts_and_passes_through():
    pyd = PydModelCallStat(
        call_type="llm",
        model="openai/x",
        prompt_tokens=3,
        completion_tokens=2,
        total_tokens=5,
    )
    orm = _to_orm_stat(pyd)
    assert isinstance(orm, ModelCallStat)
    assert orm.model == "openai/x" and orm.total_tokens == 5
    # An ORM stat is returned unchanged.
    assert _to_orm_stat(orm) is orm


@pytest.mark.asyncio
class TestPostgresTaskLogger:
    async def test_log_node_and_model_call(self, agents_session_maker, agents_db):
        storage = PostgresDataStorage.from_session_maker(agents_session_maker)
        handle = await storage.initialize_run(workflow_name="tl_wf")

        tlog = PostgresTaskLogger.from_session_maker(agents_session_maker)
        tlog.log_node(
            run_id=handle.run_id,
            session_id=handle.session_id,
            agent_id=handle.agent_id,
            node_name="classify",
            node_type="llm",
            inputs={"x": 1},
            output={"intent": "greet"},
            prompt="classify the intent",
            duration=0.5,
        )
        tlog.log_model_call(
            PydModelCallStat(call_type="llm", model="openai/x", total_tokens=5),
            agent_id=handle.agent_id,
        )
        await tlog.flush()

        async with agents_session_maker() as session:
            tasks = (await session.execute(select(Task))).scalars().all()
            assert len(tasks) == 1
            assert tasks[0].name == "classify"
            assert tasks[0].node_type == "llm"

            stats = (await session.execute(select(ModelCallStat))).scalars().all()
            assert len(stats) == 1
            assert stats[0].call_type == "llm"
            assert stats[0].total_tokens == 5

    async def test_log_node_skips_without_run(self, agents_session_maker, agents_db):
        tlog = PostgresTaskLogger.from_session_maker(agents_session_maker)
        # No run/session -> nothing can be persisted; must not raise.
        tlog.log_node(
            run_id=None,
            session_id=None,
            agent_id=None,
            node_name="start",
            node_type="start",
            inputs=None,
            output=None,
            prompt=None,
            duration=0.0,
        )
        await tlog.flush()
        async with agents_session_maker() as session:
            tasks = (await session.execute(select(Task))).scalars().all()
            assert tasks == []
