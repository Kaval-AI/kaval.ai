import os

import pytest
import pytest_asyncio
from sqlalchemy import text

AGENTS_SCHEMA = "test_agents"
os.environ["AGENTS_DB_SCHEMA"] = AGENTS_SCHEMA

from kavalai.agents.db import db_manager, Base

TEST_DB_CONFIG = dict(
    user=os.environ["AGENTS_DB_USER"],
    password=os.environ["AGENTS_DB_PASSWORD"],
    host=os.environ["AGENTS_DB_HOST"],
    port=os.environ["AGENTS_DB_PORT"],
    db_name=os.environ["AGENTS_DB_NAME"],
)


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_test_db():
    """
    Optional: Ensures the schema and tables exist before running any tests.
    """
    session_maker = db_manager.get_sessionmaker(
        user=os.environ["AGENTS_DB_USER"],
        password=os.environ["AGENTS_DB_PASSWORD"],
        host=os.environ["AGENTS_DB_HOST"],
        port=os.environ["AGENTS_DB_PORT"],
        db_name=os.environ["AGENTS_DB_NAME"],
    )

    engine = session_maker.kw["bind"]

    async with engine.begin() as conn:
        # Create schema if it doesn't exist
        await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {AGENTS_SCHEMA}"))
        # Set the search path for table creation
        await conn.execute(text(f"SET search_path TO {AGENTS_SCHEMA}"))
        # Create all tables defined in Base
        await conn.run_sync(Base.metadata.create_all)

    yield


@pytest.fixture(scope="session")
def agents_session_maker():
    return db_manager.get_sessionmaker(**TEST_DB_CONFIG)


@pytest_asyncio.fixture(scope="function")
async def agents_db(agents_session_maker):
    SessionMaker = db_manager.get_sessionmaker(**TEST_DB_CONFIG)

    async with SessionMaker() as session:
        tables = ", ".join(
            f"{AGENTS_SCHEMA}.{table.name}"
            for table in reversed(Base.metadata.sorted_tables)
        )

        if tables:
            await session.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE;"))
            await session.commit()

        yield session

        # Explicitly roll back any uncommitted changes to keep tests isolated
        await session.rollback()
