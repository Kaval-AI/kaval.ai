"""
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Built-in SQL backend for :class:`~kavalai.history.base.HistoryService`.

Stores history in the agents migration set's tables (``chat_messages``,
``tasks``, ``model_call_stats``) via the shared ORM models. Works against
Postgres and SQLite alike (the models are dialect-agnostic and schema-less;
the schema comes from the engine's ``schema_translate_map``).
"""

from typing import Dict, List, Optional
from uuid import UUID

from sqlalchemy import asc, delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kavalai.agents.db import ChatMessage, ModelCallStat, Task
from kavalai.agents.utils import clean_text, to_plain


class SqlHistoryService:
    """SQL-backed history storage (the default backend)."""

    def __init__(self, session_maker: async_sessionmaker[AsyncSession]):
        self.session_maker = session_maker

    # -- chat history --------------------------------------------------------

    async def add_chat_message(
        self,
        agent_id: UUID,
        session_id: UUID,
        role: str,
        content: Optional[str],
        run_id: Optional[UUID] = None,
    ) -> ChatMessage:
        async with self.session_maker() as session:
            message = ChatMessage(
                agent_id=agent_id,
                session_id=session_id,
                run_id=run_id,
                role=role,
                content=clean_text(content or ""),
            )
            session.add(message)
            await session.commit()
            await session.refresh(message)
            return message

    async def get_chat_messages(
        self, session_id: UUID, limit: int = 50
    ) -> List[ChatMessage]:
        async with self.session_maker() as session:
            stmt = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(asc(ChatMessage.created_at))
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    # -- task records --------------------------------------------------------

    async def add_task(
        self,
        session_id: UUID,
        run_id: UUID,
        name: Optional[str] = None,
        agent_id: Optional[UUID] = None,
        inputs: Optional[Dict] = None,
        output: Optional[Dict] = None,
        prompt: Optional[str] = None,
        errors: Optional[list[str]] = None,
        duration_seconds: Optional[float] = None,
        node_type: Optional[str] = None,
    ) -> Task:
        async with self.session_maker() as session:
            task = Task(
                agent_id=agent_id,
                session_id=session_id,
                run_id=run_id,
                name=clean_text(name),
                node_type=node_type,
                inputs=to_plain(inputs),
                output=to_plain(output),
                prompt=clean_text(prompt),
                errors=to_plain(errors),
                duration_seconds=duration_seconds,
            )
            session.add(task)
            await session.commit()
            await session.refresh(task)
            return task

    # -- model call stats ----------------------------------------------------

    async def add_model_call_stat(
        self, stats: ModelCallStat, agent_id: Optional[UUID] = None
    ) -> ModelCallStat:
        async with self.session_maker() as session:
            if agent_id:
                stats.agent_id = agent_id
            session.add(stats)
            await session.commit()
            await session.refresh(stats)
            return stats

    async def get_model_call_stats(
        self,
        call_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ModelCallStat]:
        async with self.session_maker() as session:
            stmt = (
                select(ModelCallStat)
                .order_by(ModelCallStat.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            if call_type:
                stmt = stmt.where(ModelCallStat.call_type == call_type)

            result = await session.execute(stmt)
            return list(result.scalars().all())

    # -- deletion ------------------------------------------------------------

    async def delete_for_session(self, session_id: UUID) -> None:
        async with self.session_maker() as session:
            await session.execute(
                delete(ChatMessage).where(ChatMessage.session_id == session_id)
            )
            await session.execute(delete(Task).where(Task.session_id == session_id))
            await session.commit()

    async def delete_for_agent(self, agent_id: UUID) -> None:
        async with self.session_maker() as session:
            await session.execute(
                delete(ChatMessage).where(ChatMessage.agent_id == agent_id)
            )
            await session.execute(delete(Task).where(Task.agent_id == agent_id))
            await session.execute(
                delete(ModelCallStat).where(ModelCallStat.agent_id == agent_id)
            )
            await session.commit()


# Register as a virtual subclass so isinstance checks work without inheriting
# (keeps the ABC import-light for backends that don't need it).
from kavalai.history.base import HistoryService  # noqa: E402

HistoryService.register(SqlHistoryService)
