from datetime import datetime
from uuid import UUID
from pydantic import BaseModel
from sqlalchemy import func, select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from kavalai.agents.db import Session, Run, Task, ChatMessage


class SessionSummary(BaseModel):
    session_id: UUID
    agent_id: UUID
    runs_count: int
    tasks_count: int
    messages_count: int
    first_message: str | None
    last_message: str | None
    created_at: datetime
    updated_at: datetime


async def get_sessions_summary(
    session: AsyncSession,
    agent_id: UUID | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[SessionSummary]:
    # Subquery for counts
    runs_count_sub = (
        select(Run.session_id, func.count(Run.id).label("count"))
        .group_by(Run.session_id)
        .subquery()
    )
    tasks_count_sub = (
        select(Task.session_id, func.count(Task.id).label("count"))
        .group_by(Task.session_id)
        .subquery()
    )
    messages_count_sub = (
        select(ChatMessage.session_id, func.count(ChatMessage.id).label("count"))
        .group_by(ChatMessage.session_id)
        .subquery()
    )

    # Subqueries for first and last messages
    # Using window functions might be more efficient but let's try a common approach
    # Or just fetch them in the main query if possible.
    # Actually, let's use a more direct approach for first/last messages to keep it simple and readable.

    # We'll use lateral joins or window functions if supported,
    # but for compatibility let's try separate subqueries or a combined one.

    stmt = (
        select(
            Session.id.label("session_id"),
            Session.agent_id,
            func.coalesce(runs_count_sub.c.count, 0).label("runs_count"),
            func.coalesce(tasks_count_sub.c.count, 0).label("tasks_count"),
            func.coalesce(messages_count_sub.c.count, 0).label("messages_count"),
            Session.created_at,
            Session.updated_at,
        )
        .outerjoin(runs_count_sub, Session.id == runs_count_sub.c.session_id)
        .outerjoin(tasks_count_sub, Session.id == tasks_count_sub.c.session_id)
        .outerjoin(messages_count_sub, Session.id == messages_count_sub.c.session_id)
    )

    if agent_id:
        stmt = stmt.where(Session.agent_id == agent_id)

    stmt = stmt.order_by(desc(Session.updated_at)).limit(limit).offset(offset)

    result = await session.execute(stmt)
    sessions_data = result.all()

    summaries = []
    for row in sessions_data:
        # Fetch first and last message for each session
        # This is N+1 but for a limited list it might be okay for now.
        # Alternatively we can do it in one big query with window functions.

        first_msg_stmt = (
            select(ChatMessage.content)
            .where(ChatMessage.session_id == row.session_id)
            .order_by(ChatMessage.created_at.asc())
            .limit(1)
        )
        last_msg_stmt = (
            select(ChatMessage.content)
            .where(ChatMessage.session_id == row.session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(1)
        )

        first_msg = (await session.execute(first_msg_stmt)).scalar_one_or_none()
        last_msg = (await session.execute(last_msg_stmt)).scalar_one_or_none()

        summaries.append(
            SessionSummary(
                session_id=row.session_id,
                agent_id=row.agent_id,
                runs_count=row.runs_count,
                tasks_count=row.tasks_count,
                messages_count=row.messages_count,
                first_message=first_msg,
                last_message=last_msg,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
        )

    return summaries
