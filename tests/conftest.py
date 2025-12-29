import os

import pytest_asyncio
from sqlalchemy import text

# 1. Set environment variable BEFORE importing the DB module
os.environ["POSTGRES_DB_SCHEMA"] = "kavalai_test"

from kavalai.db import AsyncKavalaiSession, Base


@pytest_asyncio.fixture(scope="function")
async def db(request):
    async with AsyncKavalaiSession() as session:
        # Get all table names from the Base metadata to truncate them
        # This is safer than 'DROP SCHEMA' as it keeps the structure
        tables = ", ".join(f"kavalai_test.{table.name}" for table in reversed(Base.metadata.sorted_tables))

        if tables:
            # TRUNCATE is much faster than DELETE for cleaning test data
            await session.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE;"))
            await session.commit()

        yield session
