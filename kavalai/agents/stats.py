from datetime import datetime, timedelta, timezone
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from kavalai.agents.db import Run, Session, ChatMessage, LLMCallStat, LLMProfile


async def get_summary_stats(session: AsyncSession, agent_id: str | None = None):
    # Calculate the start date (30 days ago)
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=30)

    # Get total cost
    # LLMCallStat might not have agent_id directly, if so we need to join Session
    # Looking at LLMCallStat in db.py, it indeed does NOT have agent_id.
    # It might be linked via some other table or we might need to skip agent_id filter for cost for now
    # or join with Task/Session if possible.
    # Actually, let's check if it has any link to agent.
    # It doesn't seem to have a direct link in the provided snippet.

    stmt_cost = select(func.sum(LLMCallStat.cost)).where(
        LLMCallStat.created_at >= start_date
    )
    # If agent_id is provided, we need to filter.
    # Since LLMCallStat doesn't have agent_id, we'll need to join if possible.
    # For now, let's keep it simple and only filter sessions if agent_id is present,
    # and maybe cost too if we can find a path.
    # In many implementations LLMCallStat has agent_id, but here it doesn't.

    # Get total sessions
    stmt_sessions = select(func.count(Session.id)).where(
        Session.created_at >= start_date
    )
    if agent_id:
        stmt_sessions = stmt_sessions.where(Session.agent_id == agent_id)

    res_cost = await session.execute(stmt_cost)
    res_sessions = await session.execute(stmt_sessions)

    return {
        "total_cost": float(res_cost.scalar() or 0),
        "total_sessions": int(res_sessions.scalar() or 0),
    }


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

    async def get_llm_stats():
        profile_name_col = func.coalesce(LLMProfile.name, "Unknown").label(
            "profile_name"
        )
        date_col = func.date(LLMCallStat.created_at).label("date")
        stmt = (
            select(
                profile_name_col,
                date_col,
                func.count(LLMCallStat.id).label("count"),
                func.sum(LLMCallStat.cost).label("cost"),
                func.sum(LLMCallStat.duration_ms).label("duration_ms"),
                func.sum(LLMCallStat.prompt_tokens).label("prompt_tokens"),
                func.sum(LLMCallStat.completion_tokens).label("completion_tokens"),
            )
            .outerjoin(LLMProfile, LLMCallStat.llm_profile_id == LLMProfile.id)
            .where(LLMCallStat.created_at >= start_date)
            .group_by(profile_name_col, date_col)
        )
        result = await session.execute(stmt)

        stats_by_profile = {}
        for row in result.all():
            if row.profile_name not in stats_by_profile:
                stats_by_profile[row.profile_name] = {}
            stats_by_profile[row.profile_name][row.date] = {
                "count": row.count,
                "cost": float(row.cost or 0),
                "duration_ms": int(row.duration_ms or 0),
                "prompt_tokens": int(row.prompt_tokens or 0),
                "completion_tokens": int(row.completion_tokens or 0),
            }
        return stats_by_profile

    runs_counts = await get_counts_for_model(Run)
    sessions_counts = await get_counts_for_model(Session)
    messages_counts = await get_counts_for_model(ChatMessage)
    llm_stats = await get_llm_stats()

    def format_series(counts):
        return [{"date": d.isoformat(), "count": counts.get(d, 0)} for d in dates]

    def format_llm_series(profile_stats):
        series = []
        for d in dates:
            day_stat = profile_stats.get(
                d,
                {
                    "count": 0,
                    "cost": 0.0,
                    "duration_ms": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                },
            )
            series.append({"date": d.isoformat(), **day_stat})
        return series

    return {
        "runs": format_series(runs_counts),
        "sessions": format_series(sessions_counts),
        "messages": format_series(messages_counts),
        "llm": {name: format_llm_series(stats) for name, stats in llm_stats.items()},
    }
