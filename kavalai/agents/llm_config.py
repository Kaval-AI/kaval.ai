import logging
import os
from datetime import datetime, timezone
from uuid import UUID

import instructor
import yaml
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kavalai.agents.db import LLMProfile
from kavalai.crud import get_all

logger = logging.getLogger(__name__)


class LLMProfileView(BaseModel):
    """LLM profile data without API key."""

    id: UUID
    name: str
    provider: str
    model_name: str
    base_url: str | None
    default_mode: str | None
    created_at: datetime
    updated_at: datetime


async def get_llm_profiles_from_db(
    session: AsyncSession,
) -> list[LLMProfileView]:
    profiles = await get_all(session, LLMProfile)
    # Sort by updated_at desc manually since crud.get_all doesn't support ordering yet
    profiles = sorted(profiles, key=lambda p: p.updated_at, reverse=True)

    return [
        LLMProfileView(
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


def load_profile_from_path(
    profile_name: str, folder_path: str = None
) -> LLMProfile | None:
    if folder_path is None:
        folder_path = os.getenv("LLM_PROFILES_PATH", "llm_profiles")
    profile_path = os.path.join(folder_path, f"{profile_name}.yaml")
    if not os.path.exists(profile_path):
        return None

    try:
        with open(profile_path, "r") as f:
            data = yaml.safe_load(f)
            return LLMProfile(**data)
    except Exception as e:
        logger.error(
            f"Error loading LLM profile '{profile_name}' from {profile_path}: {e}"
        )
        return None


async def upsert_llm_profile(session: AsyncSession, profile: LLMProfile):
    """Upsert LLM profile to the database by name."""
    stmt = select(LLMProfile).where(LLMProfile.name == profile.name)
    result = await session.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing:
        existing.provider = profile.provider
        existing.model_name = profile.model_name
        existing.api_key = profile.api_key
        existing.base_url = profile.base_url
        existing.default_mode = profile.default_mode
        existing.credentials = profile.credentials
        existing.updated_at = datetime.now(timezone.utc)
    else:
        session.add(profile)

    await session.commit()


def get_instructor(llm_profile: LLMProfile) -> instructor.Instructor:
    logger.info(
        f"Creating LLM client for profile '{llm_profile.name}': {llm_profile.provider}/{llm_profile.model_name}"
    )
    params = dict(
        model=f"{llm_profile.provider}/{llm_profile.model_name}",
        async_client=True,
        mode=instructor.Mode.JSON,
    )
    if llm_profile.base_url:
        params["base_url"] = llm_profile.base_url
    if llm_profile.api_key:
        params["api_key"] = llm_profile.api_key
    if llm_profile.default_mode:
        params["mode"] = llm_profile.default_mode
    return instructor.from_provider(**params)
