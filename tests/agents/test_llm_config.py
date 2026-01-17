import pytest
import yaml
from kavalai.agents.llm_config import (
    get_instructor,
    load_profile_from_path,
)
from kavalai.agents.db import LLMProfile, get_llm_profile_by_name, upsert_llm_profile
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
