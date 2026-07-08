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
"""

import json
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kavalai.agent_service import AgentService
from kavalai.agents.db import Run
from kavalai.workflow.state import WorkflowState
from kavalai.workflow.storage.base import ChatMsg, DataStorage, RunHandle


def _uuid(value: Optional[str]) -> Optional[UUID]:
    return UUID(value) if value else None


class PostgresDataStorage(DataStorage):
    """Postgres-backed :class:`DataStorage` delegating to :class:`AgentService`.

    This persists v2 workflow runs into the existing ``agents`` / ``sessions`` /
    ``runs`` / ``chat_messages`` tables, so the backoffice keeps showing v2 runs.
    The serialized :class:`WorkflowState` is checkpointed into ``runs.context``.
    """

    def __init__(self, agent_service: AgentService):
        self.agent_service = agent_service

    @classmethod
    def from_session_maker(
        cls, session_maker: async_sessionmaker[AsyncSession]
    ) -> "PostgresDataStorage":
        return cls(AgentService(session_maker))

    async def initialize_run(
        self,
        *,
        workflow_name: str,
        description: Optional[str] = None,
        input_schema: Optional[dict] = None,
        output_schema: Optional[dict] = None,
        workflow: Optional[dict] = None,
        session_id: Optional[str] = None,
        external_id: Optional[str] = None,
        input_data: Optional[dict] = None,
    ) -> RunHandle:
        agent, session, run = await self.agent_service.initialize_workflow_run(
            agent_name=workflow_name,
            agent_description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            workflow=workflow,
            session_id=_uuid(session_id),
            external_id=external_id,
            input_data=input_data,
        )
        return RunHandle(
            agent_id=str(agent.id),
            session_id=str(session.id),
            run_id=str(run.id),
        )

    async def update_run(
        self,
        run_id: str,
        *,
        output_data: Optional[dict] = None,
        context: Optional[dict] = None,
    ) -> None:
        await self.agent_service.update_run(
            UUID(run_id), output_data=output_data, context=context
        )

    async def save_state(self, run_id: str, state: WorkflowState) -> None:
        # Checkpoint the full state into runs.context (round-trips via load_state).
        await self.agent_service.update_run(
            UUID(run_id), context=json.loads(state.to_json())
        )

    async def load_state(self, run_id: str) -> Optional[WorkflowState]:
        async with self.agent_service.session_maker() as session:
            result = await session.execute(select(Run).where(Run.id == UUID(run_id)))
            run = result.scalar_one_or_none()
        if run is None or run.context is None:
            return None
        try:
            return WorkflowState.model_validate(run.context)
        except Exception:
            return None

    async def add_chat_message(
        self,
        *,
        agent_id: str,
        session_id: str,
        run_id: Optional[str],
        role: str,
        content: Optional[str],
    ) -> None:
        await self.agent_service.add_chat_message(
            agent_id=UUID(agent_id),
            session_id=UUID(session_id),
            role=role,
            content=content,
            run_id=_uuid(run_id),
        )

    async def get_chat_history(self, session_id: str, limit: int = 50) -> list[ChatMsg]:
        messages = await self.agent_service.get_chat_history(
            UUID(session_id), limit=limit
        )
        return [ChatMsg(role=m.role, content=m.content) for m in messages]
