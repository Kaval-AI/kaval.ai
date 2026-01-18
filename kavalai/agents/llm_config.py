import logging
import os
from datetime import datetime
from uuid import UUID

import yaml
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from kavalai import crud
from kavalai.agents.db import LLMProfile, LLMCallStat

logger = logging.getLogger(__name__)


class LLMProfileView(BaseModel):
    """LLM profile data without API key."""

    id: UUID
    name: str
    provider: str
    model_name: str
    base_url: str | None
    default_mode: str | None
    total_cost: float = 0.0
    created_at: datetime
    updated_at: datetime


async def get_llm_profiles_from_db(
    session: AsyncSession,
) -> list[LLMProfileView]:
    profiles = await crud.get_all(session, LLMProfile)

    # Fetch total cost per profile
    cost_stmt = select(
        LLMCallStat.llm_profile_id, func.sum(LLMCallStat.cost).label("total_cost")
    ).group_by(LLMCallStat.llm_profile_id)
    cost_result = await session.execute(cost_stmt)
    costs = {
        row.llm_profile_id: float(row.total_cost or 0) for row in cost_result.all()
    }

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
            total_cost=costs.get(p.id, 0.0),
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
