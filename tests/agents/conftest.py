import os

import pytest_asyncio
from sqlalchemy import text

# 1. Set environment variable BEFORE importing the DB module
AGENTS_SCHEMA = "test_agents"
os.environ["AGENTS_DB_SCHEMA"] = AGENTS_SCHEMA

from kavalai.agents.db import AsyncAgentsSession, Base


@pytest_asyncio.fixture(scope="function")
async def agents_db(request):
    async with AsyncAgentsSession() as session:
        # Get all table names from the Base metadata to truncate them
        # This is safer than 'DROP SCHEMA' as it keeps the structure
        tables = ", ".join(
            f"{AGENTS_SCHEMA}.{table.name}"
            for table in reversed(Base.metadata.sorted_tables)
        )

        if tables:
            # TRUNCATE is much faster than DELETE for cleaning test data
            await session.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE;"))
            await session.commit()

        yield session
