import pytest
import yaml
from kavalai.agents.llm_config import (
    import_llm_profiles_from_folder,
    get_instructor,
)
from kavalai.agents.db import LLMProfile
from sqlalchemy import select


@pytest.mark.asyncio
async def test_import_llm_profiles_from_folder(agents_db, tmp_path):
    # Create dummy yaml profiles
    p1 = {
        "name": "test-openai",
        "provider": "openai",
        "model_name": "gpt-4o",
    }
    p2 = {
        "name": "test-anthropic",
        "provider": "anthropic",
        "model_name": "claude-3-5-sonnet",
    }

    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()

    with open(profile_dir / "openai.yaml", "w") as f:
        yaml.dump(p1, f)
    with open(profile_dir / "anthropic.yaml", "w") as f:
        yaml.dump(p2, f)

    await import_llm_profiles_from_folder(str(profile_dir), agents_db)

    # Verify in DB
    stmt = select(LLMProfile).order_by(LLMProfile.name)
    result = await agents_db.execute(stmt)
    profiles = result.scalars().all()

    assert len(profiles) == 2
    assert profiles[0].name == "test-anthropic"
    assert profiles[1].name == "test-openai"


@pytest.mark.asyncio
async def test_get_instructor_with_auto_import(agents_db, tmp_path, monkeypatch):
    # Create dummy yaml profile
    p1 = {
        "name": "auto-imported",
        "provider": "openai/gpt-4o",
        "model_name": "gpt-4o",
    }

    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()

    with open(profile_dir / "imported.yaml", "w") as f:
        yaml.dump(p1, f)

    monkeypatch.setenv("LLM_PROFILES_PATH", str(profile_dir))

    # Try to get instructor for "auto-imported" which doesn't exist in DB yet
    client = await get_instructor("auto-imported", session=agents_db)

    assert client is not None

    # Verify it was imported to DB
    stmt = select(LLMProfile).where(LLMProfile.name == "auto-imported")
    result = await agents_db.execute(stmt)
    profile = result.scalar_one_or_none()
    assert profile is not None
    assert profile.provider == "openai/gpt-4o"


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
    client = await get_instructor(profile_name)
    assert client is not None


@pytest.mark.asyncio
async def test_get_instructor_not_found():
    with pytest.raises(Exception, match="LLM Profile 'non-existent' not found"):
        await get_instructor("non-existent")


@pytest.mark.asyncio
async def test_load_profile_from_folder_invalid_yaml(tmp_path):
    from kavalai.agents.llm_config import load_profile_from_folder

    profile_name = "invalid"
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    with open(profile_dir / f"{profile_name}.yaml", "w") as f:
        f.write("invalid: yaml: :")

    profile = load_profile_from_folder(str(profile_dir), profile_name)
    assert profile is None


@pytest.mark.asyncio
async def test_load_profile_from_folder_mismatch_name(tmp_path):
    from kavalai.agents.llm_config import load_profile_from_folder

    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    # Filename matches, but 'name' inside doesn't
    with open(profile_dir / "mismatch.yaml", "w") as f:
        yaml.dump({"name": "something-else", "provider": "openai"}, f)

    profile = load_profile_from_folder(str(profile_dir), "mismatch")
    assert profile is None
