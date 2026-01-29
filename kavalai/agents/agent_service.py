import logging
from typing import Optional, Dict, List
from uuid import UUID

from sqlalchemy import select, asc
from sqlalchemy.ext.asyncio import AsyncSession
from kavalai.agents.db import Agent, Session, Run, Task, ChatMessage, ModelCallStat

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
