from datetime import datetime, timedelta, timezone
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from kavalai.agents.db import Run, Session, ChatMessage


async def get_daily_stats(
    session: AsyncSession, days: int = 7, agent_id: str | None = None
):
    # Calculate the start date (midnight of N days ago)
    end_date = datetime.now(timezone.utc)
    start_date = (end_date - timedelta(days=days - 1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    # We want to return a list of dates from start_date to today
    dates = [(start_date + timedelta(days=i)).date() for i in range(days)]

    async def get_counts_for_model(model):
        # func.date(model.created_at) might depend on DB type,
        # for Postgres it works to get the date part.
        stmt = select(
            func.date(model.created_at).label("date"),
            func.count(model.id).label("count"),
        ).where(model.created_at >= start_date)

        if agent_id:
            if model == Run:
                # Run doesn't have agent_id, it belongs to a Session
                stmt = stmt.join(Session).where(Session.agent_id == agent_id)
            else:
                stmt = stmt.where(model.agent_id == agent_id)

        stmt = stmt.group_by(func.date(model.created_at)).order_by(
            func.date(model.created_at)
        )
        result = await session.execute(stmt)
        return {row.date: row.count for row in result.all()}

    runs_counts = await get_counts_for_model(Run)
    sessions_counts = await get_counts_for_model(Session)
    messages_counts = await get_counts_for_model(ChatMessage)

    def format_series(counts):
        return [{"date": d.isoformat(), "count": counts.get(d, 0)} for d in dates]

    return {
        "runs": format_series(runs_counts),
        "sessions": format_series(sessions_counts),
        "messages": format_series(messages_counts),
    }
