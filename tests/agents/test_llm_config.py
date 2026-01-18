from unittest.mock import AsyncMock, MagicMock
import pytest
import yaml
from pydantic import BaseModel
from kavalai.agents.llm_config import (
    get_instructor,
    load_profile_from_path,
    chat_completion_with_stats,
)
from kavalai.agents.db import (
    LLMProfile,
    get_llm_profile_by_name,
    upsert_llm_profile,
    LLMCallStat,
)
from sqlalchemy import select


@pytest.mark.asyncio
async def test_upsert_llm_profile(agents_db):
    profile = LLMProfile(
        name="test-upsert",
        provider="openai",
        model_name="gpt-4o",
    )
    await upsert_llm_profile(agents_db, profile)

    # Verify in DB
    stmt = select(LLMProfile).where(LLMProfile.name == "test-upsert")
    result = await agents_db.execute(stmt)
    db_profile = result.scalar_one()
    assert db_profile.model_name == "gpt-4o"

    # Update
    profile.model_name = "gpt-4o-mini"
    await upsert_llm_profile(agents_db, profile)

    result = await agents_db.execute(stmt)
    db_profile = result.scalar_one()
    assert db_profile.model_name == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_get_instructor_with_auto_import(agents_db, tmp_path, monkeypatch):
    # Create dummy yaml profile
    p1 = {
        "name": "auto-imported",
        "provider": "openai",
        "model_name": "gpt-4o",
    }

    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()

    with open(profile_dir / "auto-imported.yaml", "w") as f:
        yaml.dump(p1, f)

    monkeypatch.setenv("LLM_PROFILES_PATH", str(profile_dir))

    # Try to get profile for "auto-imported" which doesn't exist in DB yet
    profile = load_profile_from_path("auto-imported")
    assert profile is not None
    await upsert_llm_profile(agents_db, profile)
    client = get_instructor(profile)

    assert client is not None

    # Verify it was imported to DB
    stmt = select(LLMProfile).where(LLMProfile.name == "auto-imported")
    result = await agents_db.execute(stmt)
    profile = result.scalar_one_or_none()
    assert profile is not None
    assert profile.provider == "openai"


@pytest.mark.asyncio
async def test_get_instructor_fallback_no_session(tmp_path, monkeypatch):
    # Create dummy yaml profile
    profile_name = "manual-load"
    p1 = {
        "name": profile_name,
        "provider": "openai",
        "model_name": "gpt-4o",
    }

    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()

    # The filename must match the profile name now
    with open(profile_dir / f"{profile_name}.yaml", "w") as f:
        yaml.dump(p1, f)

    monkeypatch.setenv("LLM_PROFILES_PATH", str(profile_dir))

    # When no session is provided, it should load from the folder
    profile = load_profile_from_path(profile_name)
    assert profile is not None
    client = get_instructor(profile)
    assert client is not None


@pytest.mark.asyncio
async def test_get_llm_profile_by_name_not_found(agents_db):
    with pytest.raises(Exception, match="LLM Profile 'non-existent' not found in DB"):
        await get_llm_profile_by_name(agents_db, "non-existent")


@pytest.mark.asyncio
async def test_load_profile_from_folder_invalid_yaml(tmp_path):
    from kavalai.agents.llm_config import load_profile_from_path

    profile_name = "invalid"
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    with open(profile_dir / f"{profile_name}.yaml", "w") as f:
        f.write("invalid: yaml: :")

    profile = load_profile_from_path(profile_name, folder_path=str(profile_dir))
    assert profile is None


@pytest.mark.asyncio
async def test_load_profile_from_folder_mismatch_name(tmp_path):
    from kavalai.agents.llm_config import load_profile_from_path

    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    # Filename matches, but 'name' inside doesn't
    with open(profile_dir / "mismatch.yaml", "w") as f:
        yaml.dump({"name": "something-else", "provider": "openai"}, f)

    profile = load_profile_from_path("mismatch", folder_path=str(profile_dir))
    assert profile is not None
    assert profile.name == "something-else"


class MockResponse(BaseModel):
    message: str


@pytest.mark.asyncio
async def test_chat_completion_with_stats(agents_db, monkeypatch):
    profile = LLMProfile(
        name="test-stats",
        provider="openai",
        model_name="gpt-4o",
    )
    await upsert_llm_profile(agents_db, profile)

    # Mock instructor client
    mock_instructor = MagicMock()
    mock_create = AsyncMock()
    mock_instructor.chat.completions.create = mock_create

    # Mock response with usage
    mock_res_data = MockResponse(message="hello")
    # Attach _raw_response to the model instance
    raw_response = MagicMock()
    raw_response.usage.prompt_tokens = 10
    raw_response.usage.completion_tokens = 5
    raw_response.usage.total_tokens = 15
    raw_response.model_dump.return_value = {"id": "chat-123"}

    mock_res_data.__dict__["_raw_response"] = raw_response
    mock_create.return_value = mock_res_data

    monkeypatch.setattr(
        "kavalai.agents.llm_config.get_instructor", lambda _: mock_instructor
    )

    messages = [{"role": "user", "content": "hi"}]
    response = await chat_completion_with_stats(
        llm_profile=profile,
        response_model=MockResponse,
        messages=messages,
        session=agents_db,
    )

    assert response.message == "hello"

    # Verify stats in DB
    stmt = select(LLMCallStat).where(LLMCallStat.llm_profile_id == profile.id)
    result = await agents_db.execute(stmt)
    stat = result.scalar_one()

    assert stat.prompt_tokens == 10
    assert stat.completion_tokens == 5
    assert stat.total_tokens == 15
    assert stat.response_code == 200
    assert stat.duration_ms >= 0
    assert stat.request_data["requests"][0]["arguments"]["messages"] == messages
    assert stat.response_data == {"id": "chat-123"}


@pytest.mark.asyncio
async def test_chat_completion_with_stats_error(agents_db, monkeypatch):
    profile = LLMProfile(
        name="test-error",
        provider="openai",
        model_name="gpt-4o",
    )
    await upsert_llm_profile(agents_db, profile)

    mock_instructor = MagicMock()
    mock_create = AsyncMock(side_effect=Exception("API Error"))
    mock_instructor.chat.completions.create = mock_create

    monkeypatch.setattr(
        "kavalai.agents.llm_config.get_instructor", lambda _: mock_instructor
    )

    with pytest.raises(Exception, match="API Error"):
        await chat_completion_with_stats(
            llm_profile=profile,
            response_model=MockResponse,
            messages=[{"role": "user", "content": "hi"}],
            session=agents_db,
        )

    # Verify error stat in DB
    stmt = select(LLMCallStat).where(LLMCallStat.llm_profile_id == profile.id)
    result = await agents_db.execute(stmt)
    stat = result.scalar_one()

    assert stat.response_code == 500
    assert stat.response_data == {"error": "API Error", "error_type": "Exception"}


@pytest.mark.asyncio
async def test_chat_completion_with_stats_retry(agents_db, monkeypatch):
    profile = LLMProfile(
        name="test-retry",
        provider="openai",
        model_name="gpt-4o",
    )
    await upsert_llm_profile(agents_db, profile)

    mock_instructor = MagicMock()
    mock_create = AsyncMock()
    mock_instructor.chat.completions.create = mock_create

    # Mock first call failure, second success (if we were doing manual retries)
    # But since we currently just re-raise, we test that it captures the failure.

    mock_create.side_effect = Exception("Temporary Error")

    monkeypatch.setattr(
        "kavalai.agents.llm_config.get_instructor", lambda _: mock_instructor
    )

    with pytest.raises(Exception, match="Temporary Error"):
        await chat_completion_with_stats(
            llm_profile=profile,
            response_model=MockResponse,
            messages=[{"role": "user", "content": "hi"}],
            session=agents_db,
        )

    # Verify error stat in DB
    stmt = select(LLMCallStat).where(LLMCallStat.llm_profile_id == profile.id)
    result = await agents_db.execute(stmt)
    stat = result.scalar_one()

    assert stat.response_code == 500
    assert stat.request_data["requests"][0]["error"] == "Temporary Error"
    assert stat.request_data["requests"][0]["error_type"] == "Exception"
