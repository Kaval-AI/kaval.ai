import pytest
from uuid import uuid4
from kavalai.agents.agent_service import AgentService, load_embedding_profile_from_path
from kavalai.agents.db import LLMCallStat, LLMProfile, EmbeddingProfile


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

    async def test_llm_profiles_logic(self, agents_db):
        service = AgentService(agents_db)

        # 1. Test upsert new profile
        profile = LLMProfile(
            name="TestLLM",
            provider="openai",
            model_name="gpt-4",
            api_key="key1",
            credentials={"some": "cred"},
        )
        saved = await service.upsert_llm_profile(profile)
        assert saved.id is not None
        assert saved.name == "TestLLM"

        # 2. Test get by name
        retrieved = await service.get_llm_profile_by_name("TestLLM")
        assert retrieved.id == saved.id

        # 3. Test upsert existing (update)
        profile.api_key = "key2"
        updated = await service.upsert_llm_profile(profile)
        assert updated.id == saved.id
        assert updated.api_key == "key2"

        # 4. Test get all views
        views = await service.get_llm_profiles_from_db()
        assert len(views) == 1
        assert views[0].name == "TestLLM"

        # 5. Test get by name - not found
        with pytest.raises(Exception) as excinfo:
            await service.get_llm_profile_by_name("NonExistent")
        assert "not found" in str(excinfo.value)

    async def test_load_llm_profile_from_path(self, tmp_path):
        from kavalai.agents.agent_service import load_profile_from_path

        # Create a dummy yaml profile
        d = tmp_path / "llm_profiles"
        d.mkdir()
        p = d / "test_llm_file.yaml"
        p.write_text(
            "name: test_llm_file\nprovider: openai\nmodel_name: gpt-4\napi_key: file-key\n"
        )

        # Test loading
        profile = load_profile_from_path("test_llm_file", folder_path=str(d))
        assert profile is not None
        assert profile.name == "test_llm_file"
        assert profile.api_key == "file-key"

        # Test loading non-existent
        assert load_profile_from_path("ghost", folder_path=str(d)) is None

    async def test_embedding_profiles_logic(self, agents_db):
        service = AgentService(agents_db)

        # 1. Test upsert new profile
        profile = EmbeddingProfile(
            name="TestEmbed",
            provider="openai",
            model_name="text-embedding-3-small",
            api_key="key1",
            credentials={"some": "cred"},
        )
        saved = await service.upsert_embedding_profile(profile)
        assert saved.id is not None
        assert saved.name == "TestEmbed"

        # 2. Test get by name
        retrieved = await service.get_embedding_profile_by_name("TestEmbed")
        assert retrieved.id == saved.id

        # 3. Test upsert existing (update)
        profile.api_key = "key2"
        updated = await service.upsert_embedding_profile(profile)
        assert updated.id == saved.id
        assert updated.api_key == "key2"

        # 4. Test get all views
        views = await service.get_embedding_profiles_from_db()
        assert len(views) == 1
        assert views[0].name == "TestEmbed"
        assert not hasattr(
            views[0], "api_key"
        )  # LLMEmbeddingView should not have api_key

        # 5. Test get by name - not found
        with pytest.raises(Exception) as excinfo:
            await service.get_embedding_profile_by_name("NonExistent")
        assert "not found" in str(excinfo.value)

    async def test_load_embedding_profile_from_path(self, tmp_path, monkeypatch):
        # Create a dummy yaml profile
        d = tmp_path / "embedding_profiles"
        d.mkdir()
        p = d / "test_embed_file.yaml"
        p.write_text(
            "name: test_embed_file\nprovider: openai\nmodel_name: text-embedding-3-small\napi_key: file-key\n"
        )

        # Test loading with explicit path
        profile = load_embedding_profile_from_path(
            "test_embed_file", folder_path=str(d)
        )
        assert profile is not None
        assert profile.name == "test_embed_file"
        assert profile.api_key == "file-key"

        # Test loading with env var
        monkeypatch.setenv("EMBEDDING_PROFILES_PATH", str(d))
        profile = load_embedding_profile_from_path("test_embed_file")
        assert profile is not None
        assert profile.name == "test_embed_file"

        # Test loading non-existent
        assert load_embedding_profile_from_path("ghost", folder_path=str(d)) is None
