import pytest
from uuid import uuid4
from kavalai.agents.agent_service import AgentService
from kavalai.agents.db import ModelCallStat


@pytest.mark.asyncio
class TestAgentService:
    async def test_get_or_create_agent(self, agents_db):
        service = AgentService(agents_db)

        # Test Creation
        agent = await service.get_or_create_agent(
            name="ResearchAgent",
            description="Tests the agent creation",
            workflow={"steps": ["start", "end"]},
        )
        assert agent.name == "ResearchAgent"
        assert agent.workflow["steps"] == ["start", "end"]

        # Test Retrieval
        existing_agent = await service.get_or_create_agent(name="ResearchAgent")
        assert existing_agent.id == agent.id

    async def test_get_or_create_session_logic(self, agents_db):
        service = AgentService(agents_db)
        agent = await service.get_or_create_agent(name="SessionTest")

        # 1. Test creation when no session_id is provided
        session = await service.get_or_create_session(agent_id=agent.id)
        assert session.id is not None

        # 2. Test retrieval with existing session_id
        retrieved = await service.get_or_create_session(
            agent_id=agent.id, session_id=session.id
        )
        assert retrieved.id == session.id

        # 3. Test non-existent session_id returns None
        not_found = await service.get_or_create_session(
            agent_id=agent.id, session_id=uuid4()
        )
        assert not_found is None

    async def test_run_and_task_tracking(self, agents_db):
        service = AgentService(agents_db)
        agent = await service.get_or_create_agent(name="TaskTest")
        session = await service.get_or_create_session(agent_id=agent.id)

        # Create Run
        run = await service.create_run(
            session_id=session.id, input_data={"user_query": "search for AI news"}
        )
        assert run.id is not None

        # Add Task to Run
        task = await service.add_task(
            session_id=session.id,
            run_id=run.id,
            agent_id=agent.id,
            inputs={"query": "AI news"},
            output={"results": ["result1"]},
        )
        assert task.run_id == run.id
        assert task.output["results"] == ["result1"]

    async def test_chat_history_retrieval(self, agents_db):
        service = AgentService(agents_db)
        agent = await service.get_or_create_agent(name="ChatTest")
        session = await service.get_or_create_session(agent_id=agent.id)

        # Add a few messages
        await service.add_chat_message(agent.id, session.id, "user", "Message 1")
        await service.add_chat_message(agent.id, session.id, "assistant", "Response 1")

        history = await service.get_chat_history(session.id)

        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "Message 1"
        assert history[1].role == "assistant"
        assert history[1].content == "Response 1"

    async def test_get_model_call_stats(self, agents_db):
        service = AgentService(agents_db)

        # Create some call stats
        for i in range(10):
            stat = ModelCallStat(
                call_type="llm",
                model="gpt-4o",
                response_code=200,
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                duration_seconds=0.1,
                request_data={"query": f"test {i}"},
                response_data={"answer": f"result {i}"},
                cost=0.001,
            )
            agents_db.add(stat)
        await agents_db.commit()

        # Test retrieval all
        stats = await service.get_model_call_stats()
        assert len(stats) == 10

        # Test filter by type
        stats = await service.get_model_call_stats(call_type="llm")
        assert len(stats) == 10

        # Test filter by non-existent type
        stats = await service.get_model_call_stats(call_type="embedding")
        assert len(stats) == 0

        # Test pagination
        stats = await service.get_model_call_stats(limit=5, offset=0)
        assert len(stats) == 5

        stats = await service.get_model_call_stats(limit=5, offset=5)
        assert len(stats) == 5

    async def test_get_history_value(self, agents_db):
        service = AgentService(agents_db)
        agent = await service.get_or_create_agent(name="HistoryTest")
        session = await service.get_or_create_session(agent_id=agent.id)

        # Create Run 1
        run1 = await service.create_run(session_id=session.id)
        await service.update_run(
            run_id=run1.id,
            context={"search_results": "result1", "other": "val1", "nested": {"a": 1}},
        )

        # Create Run 2
        run2 = await service.create_run(session_id=session.id)
        await service.update_run(
            run_id=run2.id,
            context={
                "search_results": "result2",
                "something": "else",
                "nested": {"b": 2},
            },
        )

        # Test single retrieval (most recent)
        val = await service.get_history_value(session.id, "search_results")
        assert val == "result2"

        val_other = await service.get_history_value(session.id, "other")
        assert val_other == "val1"

        # Test path retrieval
        val_nested = await service.get_history_value(session.id, "nested.b")
        assert val_nested == 2

        # Test non-existent key
        val_none = await service.get_history_value(session.id, "non_existent")
        assert val_none is None
