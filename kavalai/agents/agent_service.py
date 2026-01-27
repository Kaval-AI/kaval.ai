import os
from datetime import datetime, timezone
from typing import Optional, Dict, List
from uuid import UUID

import yaml
from pydantic import BaseModel
from sqlalchemy import asc
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from kavalai import crud
from kavalai.agents.db import Agent, Session, Run, Task, ChatMessage
from kavalai.agents.db import (
    LLMProfile,
    LLMCallStat,
    EmbeddingProfile,
    EmbeddingCallStat,
)

logger = logging.getLogger(__name__)


class LLMProfileView(BaseModel):
    """LLM profile data without API key."""

    id: UUID
    name: str
    provider: str
    model_name: str
    base_url: str | None
    config: dict | None
    total_cost: float = 0.0
    created_at: datetime
    updated_at: datetime


class LLMEmbeddingView(BaseModel):
    """Embedding profile data without API key."""

    id: UUID
    name: str
    provider: str
    model_name: str
    base_url: str | None
    embedding_size: int | None
    config: dict | None
    total_cost: float = 0.0
    created_at: datetime
    updated_at: datetime


class AgentService:
    """Provider common database operation for Agents."""

    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    async def get_or_create_agent(
        self,
        name: str,
        description: Optional[str] = None,
        input_schema: Optional[Dict] = None,
        output_schema: Optional[Dict] = None,
        workflow: Optional[Dict] = None,
    ) -> Agent:
        """Finds an agent by name or creates a new one if not found."""
        stmt = select(Agent).where(Agent.name == name)
        result = await self.db.execute(stmt)
        agent = result.scalar_one_or_none()

        if not agent:
            agent = Agent(
                name=name,
                description=description,
                input_schema=input_schema,
                output_schema=output_schema,
                workflow=workflow,
            )
            self.db.add(agent)
            await self.db.commit()
            await self.db.refresh(agent)
        else:
            # Update existing agent if description, schemas or workflow have changed
            # We only update if the new value is not None to avoid accidental overwrites
            updates = {
                "description": description,
                "input_schema": input_schema,
                "output_schema": output_schema,
                "workflow": workflow,
            }
            logger.info(f"UPDATES: {updates}")
            changed = False
            for field, value in updates.items():
                if value is not None and getattr(agent, field) != value:
                    setattr(agent, field, value)
                    changed = True
            logger.info(f"CHANGED: {changed}")
            if changed:
                await self.db.commit()
                await self.db.refresh(agent)

        return agent

    async def get_or_create_session(
        self,
        agent_id: UUID,
        session_id: Optional[UUID] = None,
        external_id: Optional[UUID] = None,
    ) -> Optional[Session]:
        if session_id:
            stmt = select(Session).where(Session.id == session_id)
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()

        # No session_id provided? Create a new one.
        new_session = Session(agent_id=agent_id, external_id=external_id)
        self.db.add(new_session)
        await self.db.commit()
        await self.db.refresh(new_session)
        return new_session

    async def create_run(
        self,
        session_id: UUID,
        input_data: Optional[Dict] = None,
        context: Optional[Dict] = None,
    ) -> Run:
        """Creates a new run entry for a specific session."""
        run = Run(session_id=session_id, input_data=input_data, context=context)
        self.db.add(run)
        await self.db.commit()
        await self.db.refresh(run)
        return run

    async def add_task(
        self,
        session_id: UUID,
        run_id: UUID,
        agent_id: Optional[UUID] = None,
        inputs: Optional[Dict] = None,
        output: Optional[Dict] = None,
    ) -> Task:
        """Records a specific unit of work (Task) performed within a run."""
        task = Task(
            agent_id=agent_id,
            session_id=session_id,
            run_id=run_id,
            inputs=inputs,
            output=output,
        )
        self.db.add(task)
        await self.db.commit()
        await self.db.refresh(task)
        return task

    async def get_chat_history(
        self, session_id: UUID, limit: int = 50
    ) -> List[ChatMessage]:
        """
        Retrieves the conversation history for a session,
        ordered from oldest to newest.
        """
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(asc(ChatMessage.created_at))
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def add_chat_message(
        self,
        agent_id: UUID,
        session_id: UUID,
        role: str,
        content: str,
        run_id: Optional[UUID] = None,
    ) -> ChatMessage:
        """Helper to append messages to the chat history."""
        message = ChatMessage(
            agent_id=agent_id,
            session_id=session_id,
            run_id=run_id,
            role=role,
            content=content,
        )
        self.db.add(message)
        await self.db.commit()
        await self.db.refresh(message)
        return message

    async def get_llm_profile_by_name(self, profile_name: str) -> LLMProfile:
        """
        Get an LLM profile from DB by name.
        """
        stmt = select(LLMProfile).where(LLMProfile.name == profile_name)
        result = await self.db.execute(stmt)
        profile = result.scalar_one_or_none()

        if not profile:
            raise Exception(f"LLM Profile '{profile_name}' not found in DB")

        return profile

    async def upsert_llm_profile(self, profile: LLMProfile) -> LLMProfile:
        """Upsert LLM profile to the database by name and return the profile with ID."""
        stmt = select(LLMProfile).where(LLMProfile.name == profile.name)
        result = await self.db.execute(stmt)
        existing = result.scalar_one_or_none()

        profile_data = {
            "name": profile.name,
            "provider": profile.provider,
            "model_name": profile.model_name,
            "api_key": profile.api_key,
            "base_url": profile.base_url,
            "config": profile.config,
            "updated_at": datetime.now(timezone.utc),
        }

        if existing:
            return await crud.update(self.db, LLMProfile, existing.id, profile_data)
        else:
            return await crud.insert(self.db, LLMProfile, profile_data)

    async def get_llm_profiles_from_db(self) -> list[LLMProfileView]:
        profiles = await crud.get_all(self.db, LLMProfile)

        # Fetch total cost per profile
        cost_stmt = select(
            LLMCallStat.llm_profile_id, func.sum(LLMCallStat.cost).label("total_cost")
        ).group_by(LLMCallStat.llm_profile_id)
        cost_result = await self.db.execute(cost_stmt)
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
                config={
                    k: v
                    for k, v in (p.config or {}).items()
                    if "key" not in k.lower() and "secret" not in k.lower()
                },
                total_cost=costs.get(p.id, 0.0),
                created_at=p.created_at,
                updated_at=p.updated_at,
            )
            for p in profiles
        ]

    async def get_embedding_profile_by_name(
        self, profile_name: str
    ) -> EmbeddingProfile:
        """
        Get an embedding profile from DB by name.
        """
        stmt = select(EmbeddingProfile).where(EmbeddingProfile.name == profile_name)
        result = await self.db.execute(stmt)
        profile = result.scalar_one_or_none()

        if not profile:
            raise Exception(f"Embedding Profile '{profile_name}' not found in DB")

        return profile

    async def upsert_embedding_profile(
        self, profile: EmbeddingProfile
    ) -> EmbeddingProfile:
        """Upsert embedding profile to the database by name and return the profile with ID."""
        stmt = select(EmbeddingProfile).where(EmbeddingProfile.name == profile.name)
        result = await self.db.execute(stmt)
        existing = result.scalar_one_or_none()

        profile_data = {
            "name": profile.name,
            "provider": profile.provider,
            "model_name": profile.model_name,
            "api_key": profile.api_key,
            "base_url": profile.base_url,
            "embedding_size": profile.embedding_size,
            "config": profile.config,
            "updated_at": datetime.now(timezone.utc),
        }

        if existing:
            return await crud.update(
                self.db, EmbeddingProfile, existing.id, profile_data
            )
        else:
            return await crud.insert(self.db, EmbeddingProfile, profile_data)

    async def get_embedding_profiles_from_db(self) -> list[LLMEmbeddingView]:
        profiles = await crud.get_all(self.db, EmbeddingProfile)

        # Fetch total cost per profile
        cost_stmt = select(
            EmbeddingCallStat.embedding_profile_id,
            func.sum(EmbeddingCallStat.cost).label("total_cost"),
        ).group_by(EmbeddingCallStat.embedding_profile_id)
        cost_result = await self.db.execute(cost_stmt)
        costs = {
            row.embedding_profile_id: float(row.total_cost or 0)
            for row in cost_result.all()
        }

        # Sort by updated_at desc manually since crud.get_all doesn't support ordering yet
        profiles = sorted(profiles, key=lambda p: p.updated_at, reverse=True)

        return [
            LLMEmbeddingView(
                id=p.id,
                name=p.name,
                provider=p.provider,
                model_name=p.model_name,
                base_url=p.base_url,
                embedding_size=p.embedding_size,
                config={
                    k: v
                    for k, v in (p.config or {}).items()
                    if "key" not in k.lower() and "secret" not in k.lower()
                },
                total_cost=costs.get(p.id, 0.0),
                created_at=p.created_at,
                updated_at=p.updated_at,
            )
            for p in profiles
        ]

    async def get_llm_call_stats(
        self,
        llm_profile_id: Optional[UUID] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[LLMCallStat]:
        """
        Retrieves paginated LLM call stats, optionally filtered by LLM profile.
        """
        stmt = (
            select(LLMCallStat)
            .order_by(LLMCallStat.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if llm_profile_id:
            stmt = stmt.where(LLMCallStat.llm_profile_id == llm_profile_id)

        result = await self.db.execute(stmt)
        return list(result.scalars().all())


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


def load_embedding_profile_from_path(
    profile_name: str, folder_path: str = None
) -> EmbeddingProfile | None:
    if folder_path is None:
        folder_path = os.getenv("EMBEDDING_PROFILES_PATH", "embedding_profiles")
    profile_path = os.path.join(folder_path, f"{profile_name}.yaml")
    if not os.path.exists(profile_path):
        return None

    try:
        with open(profile_path, "r") as f:
            data = yaml.safe_load(f)
            return EmbeddingProfile(**data)
    except Exception as e:
        logger.error(
            f"Error loading embedding profile '{profile_name}' from {profile_path}: {e}"
        )
        return None
