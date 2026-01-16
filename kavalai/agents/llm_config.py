import os
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
import yaml
import instructor
from kavalai.agents.db import LLMProfile


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
    stmt = select(LLMProfile).order_by(desc(LLMProfile.updated_at))
    result = await session.execute(stmt)
    profiles = result.scalars().all()

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

                profile_data = LLMProfileCreate(**data)
                # Check if profile exists by name
                stmt = select(LLMProfile).where(LLMProfile.name == profile_data.name)
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()

                if existing:
                    existing.provider = profile_data.provider
                    existing.model_name = profile_data.model_name
                    existing.api_key = profile_data.api_key
                    existing.base_url = profile_data.base_url
                    existing.default_mode = profile_data.default_mode
                    existing.credentials = profile_data.credentials
                else:
                    new_profile = LLMProfile(
                        name=profile_data.name,
                        provider=profile_data.provider,
                        model_name=profile_data.model_name,
                        api_key=profile_data.api_key,
                        base_url=profile_data.base_url,
                        default_mode=profile_data.default_mode,
                        credentials=profile_data.credentials,
                    )
                    session.add(new_profile)

    await session.commit()


def load_profile_from_folder(
    folder_path: str, profile_name: str
) -> LLMProfileCreate | None:
    if not os.path.exists(folder_path):
        return None

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            with open(os.path.join(folder_path, filename), "r") as f:
                try:
                    data = yaml.safe_load(f)
                except yaml.YAMLError:
                    continue
                if not data or "name" not in data:
                    continue

                if data["name"] == profile_name:
                    return LLMProfileCreate(**data)
    return None


async def get_instructor(
    profile_name: str, session: AsyncSession | None = None
) -> instructor.Instructor:
    async def lookup_profile() -> LLMProfile:
        stmt = select(LLMProfile).where(LLMProfile.name == profile_name)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    profile = None
    if session:
        profile = await lookup_profile()
        if not profile:
            # Try to import from folder if not found
            profiles_path = os.getenv("LLM_PROFILES_PATH", "llm_profiles")
            await import_llm_profiles_from_folder(profiles_path, session)
            profile = await lookup_profile()

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
