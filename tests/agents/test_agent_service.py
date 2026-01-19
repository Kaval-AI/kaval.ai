import pytest
from uuid import uuid4
from kavalai.agents.agent_service import AgentService
from kavalai.agents.db import LLMCallStat, LLMProfile


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

    async def test_get_llm_call_stats(self, agents_db):
        service = AgentService(agents_db)

        # Create an LLM profile
        profile = LLMProfile(
            name="TestProfile",
            provider="openai",
            model_name="gpt-4o",
            api_key="fake-key",
        )
        agents_db.add(profile)
        await agents_db.commit()
        await agents_db.refresh(profile)

        # Create some call stats
        for i in range(10):
            stat = LLMCallStat(
                llm_profile_id=profile.id,
                name=f"Call {i}",
                response_code=200,
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                duration_ms=100,
                request_data={"query": f"test {i}"},
                response_data={"answer": f"result {i}"},
                cost=0.001,
            )
            agents_db.add(stat)
        await agents_db.commit()

        # Test retrieval all
        stats = await service.get_llm_call_stats()
        assert len(stats) == 10

        # Test filter by profile
        stats = await service.get_llm_call_stats(llm_profile_id=profile.id)
        assert len(stats) == 10

        # Test filter by non-existent profile
        stats = await service.get_llm_call_stats(llm_profile_id=uuid4())
        assert len(stats) == 0

        # Test pagination
        stats = await service.get_llm_call_stats(limit=5, offset=0)
        assert len(stats) == 5
        assert stats[0].name == "Call 9"  # Ordered by created_at desc

        stats = await service.get_llm_call_stats(limit=5, offset=5)
        assert len(stats) == 5
        assert stats[0].name == "Call 4"
