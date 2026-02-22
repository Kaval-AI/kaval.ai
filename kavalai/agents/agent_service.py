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

import logging
from typing import Optional, Dict, List, Any
from uuid import UUID

from sqlalchemy import select, asc
from sqlalchemy.ext.asyncio import AsyncSession
from kavalai.agents.db import Agent, Session, Run, Task, ChatMessage, ModelCallStat
from kavalai.agents.resolvers import resolve_path, find_key_recursive

logger = logging.getLogger(__name__)


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
            changed = False
            for field, value in updates.items():
                if value is not None and getattr(agent, field) != value:
                    setattr(agent, field, value)
                    changed = True
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

    async def update_run(
        self,
        run_id: UUID,
        *,
        output_data: Optional[Dict] = None,
        context: Optional[Dict] = None,
    ) -> Run:
        """Updates an existing run with final output_data and/or context."""
        stmt = select(Run).where(Run.id == run_id)
        result = await self.db.execute(stmt)
        run = result.scalar_one_or_none()
        if not run:
            raise ValueError(f"Run not found: {run_id}")
        if output_data is not None:
            run.output_data = output_data
        if context is not None:
            run.context = context
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

    async def get_history_value(self, session_id: UUID, key: str) -> Optional[Any]:
        """
        Retrieves a value from the context of previous runs in the same session.

        - If `key` is a dotted path (e.g., "output.search_results"), resolves it as such.
        - If `key` is a plain name (e.g., "search_results"), searches recursively for the
          first matching key in the context dicts of previous runs (newest first).

        Returns the most recent value found for the given key.
        """
        is_path = "." in key

        # Fetch recent contexts for the session, newest first, and scan for the key.
        stmt = (
            select(Run.context)
            .where(Run.session_id == session_id)
            .order_by(Run.created_at.desc())
        )
        result = await self.db.execute(stmt)
        rows = list(result.scalars().all())

        for row in rows:
            if not row:
                continue
            if is_path:
                val = resolve_path(row, key)
            else:
                val = find_key_recursive(row, key)
            if val is not None:
                return val

        return None

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

    async def get_model_call_stats(
        self,
        call_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ModelCallStat]:
        """
        Retrieves paginated model call stats, optionally filtered by call type.
        """
        stmt = (
            select(ModelCallStat)
            .order_by(ModelCallStat.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if call_type:
            stmt = stmt.where(ModelCallStat.call_type == call_type)

        result = await self.db.execute(stmt)
        return list(result.scalars().all())
