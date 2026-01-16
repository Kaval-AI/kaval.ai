import os
from datetime import datetime
from uuid import UUID

import instructor
import yaml
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from kavalai.crud import insert, update, get_all
from kavalai.agents.db import LLMProfile
import logging

logger = logging.getLogger(__name__)


class LLMProfileCreate(BaseModel):
    name: str
    provider: str
    model_name: str
    api_key: str | None = None
    base_url: str | None = None
    default_mode: str | None = None
    credentials: dict = {}


class LLMProfileRead(BaseModel):
    id: UUID
    name: str
    provider: str
    model_name: str
    base_url: str | None
    default_mode: str | None
    created_at: datetime
    updated_at: datetime


async def get_llm_profiles(
    session: AsyncSession,
) -> list[LLMProfileRead]:
    profiles = await get_all(session, LLMProfile)
    # Sort by updated_at desc manually since crud.get_all doesn't support ordering yet
    profiles = sorted(profiles, key=lambda p: p.updated_at, reverse=True)

    return [
        LLMProfileRead(
            id=p.id,
            name=p.name,
            provider=p.provider,
            model_name=p.model_name,
            base_url=p.base_url,
            default_mode=p.default_mode,
            created_at=p.created_at,
            updated_at=p.updated_at,
        )
        for p in profiles
    ]


async def import_llm_profiles_from_folder(
    folder_path: str, session: AsyncSession
) -> None:
    if not os.path.exists(folder_path):
        return

    # Keep track of profiles we've updated in this call to avoid redundant DB operations
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            with open(os.path.join(folder_path, filename), "r") as f:
                try:
                    data = yaml.safe_load(f)
                except yaml.YAMLError:
                    continue
                if not data or "name" not in data:
                    continue

                existing = await lookup_profile(data["name"], session)

                if existing:
                    # Update existing record using crud.update
                    await update(session, LLMProfile, existing.id, data)
                else:
                    # Insert new record using crud.insert
                    await insert(session, LLMProfile, data)


def load_profile_from_folder(folder_path: str, profile_name: str) -> LLMProfile | None:
    if not os.path.exists(folder_path):
        return None
    profile_path = os.path.join(folder_path, f"{profile_name}.yaml")
    if os.path.exists(profile_path):
        with open(profile_path, "r") as f:
            try:
                data = yaml.safe_load(f)
                if data and isinstance(data, dict):
                    # Ensure the name in the file matches what we expect
                    if data.get("name") == profile_name:
                        logger.info(
                            f"Loaded LLM profile '{profile_name}' from {profile_path}"
                        )
                        return LLMProfile(**data)
            except yaml.YAMLError:
                pass
    logger.warning(f"Could not load '{profile_name}' from {profile_path}.")
    return None


async def lookup_profile(profile_name: str, session: AsyncSession) -> LLMProfile:
    result = await session.execute(
        select(LLMProfile).where(LLMProfile.name == profile_name)
    )
    result = result.scalar_one_or_none()
    if result:
        logger.info(f"Found LLM profile '{profile_name}' in DB")
    else:
        logger.info(f"LLM profile '{profile_name}' not found in DB")
    return result


async def get_instructor(
    profile_name: str, session: AsyncSession | None = None
) -> instructor.Instructor:
    profile = None
    if session:
        profile = await lookup_profile(profile_name, session)
        if not profile:
            # Try to import from folder if not found and session is available
            profiles_path = os.getenv("LLM_PROFILES_PATH", "llm_profiles")
            await import_llm_profiles_from_folder(profiles_path, session)
            profile = await lookup_profile(profile_name, session)

    if not profile:
        # Try to load from folder directly if no session or not found in DB
        profiles_path = os.getenv("LLM_PROFILES_PATH", "llm_profiles")
        profile = load_profile_from_folder(profiles_path, profile_name)

    if not profile:
        raise Exception(f"LLM Profile '{profile_name}' not found")

    params = dict(
        model=f"{profile.provider}/{profile.model_name}",
        async_client=True,
        mode=instructor.Mode.JSON,
    )

    if profile.base_url:
        params["base_url"] = profile.base_url
    if profile.api_key:
        params["api_key"] = profile.api_key

    return instructor.from_provider(**params)
