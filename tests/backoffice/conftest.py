import os

import pytest_asyncio
from sqlalchemy import text

# 1. Set environment variable BEFORE importing the DB module
BACKOFFICE_SCHEMA = "test_backoffice"
os.environ["BACKOFFICE_DB_SCHEMA"] = BACKOFFICE_SCHEMA

from kavalai.backoffice.db import AsyncBackofficeSession, Base


@pytest_asyncio.fixture(scope="function")
async def backoffice_db(request):
    async with AsyncBackofficeSession() as session:
        # Get all table names from the Base metadata to truncate them
        # This is safer than 'DROP SCHEMA' as it keeps the structure
        tables = ", ".join(
            f"{BACKOFFICE_SCHEMA}.{table.name}"
            for table in reversed(Base.metadata.sorted_tables)
        )

        if tables:
            # TRUNCATE is much faster than DELETE for cleaning test data
            await session.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE;"))
            await session.commit()

        yield session
