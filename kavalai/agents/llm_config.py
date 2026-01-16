from datetime import datetime
from uuid import UUID
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from kavalai.agents.db import LLMProfile


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
