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

from datetime import datetime, timedelta, timezone
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from kavalai.agents.db import (
    Run,
    Session,
    ChatMessage,
    ModelCallStat,
)


async def get_summary_stats(session: AsyncSession, agent_id: str | None = None):
    # Calculate the start date (30 days ago)
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=30)

    # Get LLM cost
    stmt_llm_cost = select(func.sum(ModelCallStat.cost)).where(
        ModelCallStat.call_type == "llm", ModelCallStat.created_at >= start_date
    )
    if agent_id:
        stmt_llm_cost = stmt_llm_cost.where(ModelCallStat.agent_id == agent_id)

    # Get Embedding cost
    stmt_embedding_cost = select(func.sum(ModelCallStat.cost)).where(
        ModelCallStat.call_type == "embedding", ModelCallStat.created_at >= start_date
    )
    if agent_id:
        stmt_embedding_cost = stmt_embedding_cost.where(
            ModelCallStat.agent_id == agent_id
        )

    # Get total sessions
    stmt_sessions = select(func.count(Session.id)).where(
        Session.created_at >= start_date
    )
    if agent_id:
        stmt_sessions = stmt_sessions.where(Session.agent_id == agent_id)

    # Get total tokens
    stmt_tokens = select(
        func.sum(ModelCallStat.prompt_tokens), func.sum(ModelCallStat.completion_tokens)
    ).where(ModelCallStat.call_type == "llm", ModelCallStat.created_at >= start_date)
    if agent_id:
        stmt_tokens = stmt_tokens.where(ModelCallStat.agent_id == agent_id)

    res_llm_cost = await session.execute(stmt_llm_cost)
    res_embedding_cost = await session.execute(stmt_embedding_cost)
    res_sessions = await session.execute(stmt_sessions)
    res_tokens = await session.execute(stmt_tokens)

    llm_cost = float(res_llm_cost.scalar() or 0)
    embedding_cost = float(res_embedding_cost.scalar() or 0)
    tokens_row = res_tokens.one()

    return {
        "total_cost": llm_cost + embedding_cost,
        "llm_cost": llm_cost,
        "embedding_cost": embedding_cost,
        "total_sessions": int(res_sessions.scalar() or 0),
        "total_prompt_tokens": int(tokens_row[0] or 0),
        "total_completion_tokens": int(tokens_row[1] or 0),
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
        model_name_col = ModelCallStat.model.label("model_name")
        date_col = func.date(ModelCallStat.created_at).label("date")
        stmt = select(
            model_name_col,
            date_col,
            func.count(ModelCallStat.id).label("count"),
            func.sum(ModelCallStat.cost).label("cost"),
            func.sum(ModelCallStat.duration_seconds).label("duration_seconds"),
            func.sum(ModelCallStat.prompt_tokens).label("prompt_tokens"),
            func.sum(ModelCallStat.completion_tokens).label("completion_tokens"),
        ).where(
            ModelCallStat.call_type == "llm", ModelCallStat.created_at >= start_date
        )
        if agent_id:
            stmt = stmt.where(ModelCallStat.agent_id == agent_id)

        stmt = stmt.group_by(model_name_col, date_col)
        result = await session.execute(stmt)

        stats_by_model = {}
        for row in result.all():
            if row.model_name not in stats_by_model:
                stats_by_model[row.model_name] = {}
            stats_by_model[row.model_name][row.date] = {
                "count": row.count,
                "cost": float(row.cost or 0),
                "duration_seconds": float(row.duration_seconds or 0),
                "prompt_tokens": int(row.prompt_tokens or 0),
                "completion_tokens": int(row.completion_tokens or 0),
            }
        return stats_by_model

    runs_counts = await get_counts_for_model(Run)
    sessions_counts = await get_counts_for_model(Session)
    messages_counts = await get_counts_for_model(ChatMessage)
    llm_stats = await get_llm_stats()

    def format_series(counts):
        return [{"date": d.isoformat(), "count": counts.get(d, 0)} for d in dates]

    def format_llm_series(model_stats):
        series = []
        for d in dates:
            day_stat = model_stats.get(
                d,
                {
                    "count": 0,
                    "cost": 0.0,
                    "duration_seconds": 0.0,
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
