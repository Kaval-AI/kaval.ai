import os
import pytest
import pytest_asyncio
from sqlalchemy import text
from testcontainers.postgres import PostgresContainer
from kavalai.migrate_db import migrate
from kavalai import SQL_MIGRATIONS_PATH

AGENTS_SCHEMA = "test_agents"
os.environ["AGENTS_DB_SCHEMA"] = AGENTS_SCHEMA

from kavalai.agents.db import db_manager, Base


@pytest.fixture(scope="session")
def postgres_container():
    with PostgresContainer("pgvector/pgvector:pg15-trixie") as postgres:
        yield postgres


@pytest.fixture(scope="session")
def agents_db_config(postgres_container):
    config = dict(
        user=postgres_container.username,
        password=postgres_container.password,
        host=postgres_container.get_container_host_ip(),
        port=int(postgres_container.get_exposed_port(5432)),
        db_name=postgres_container.dbname,
    )
    # Set environment variables for the SDK to pick up
    os.environ["AGENTS_DB_USER"] = config["user"]
    os.environ["AGENTS_DB_PASSWORD"] = config["password"]
    os.environ["AGENTS_DB_HOST"] = config["host"]
    os.environ["AGENTS_DB_PORT"] = str(config["port"])
    os.environ["AGENTS_DB_NAME"] = config["db_name"]
    os.environ["AGENTS_DB_SCHEMA"] = AGENTS_SCHEMA
    return config


@pytest.fixture(scope="session")
def migrated_agents_db(agents_db_config):
    migrations_dir = os.path.join(SQL_MIGRATIONS_PATH, "app")
    migrate(
        migrations_dir=migrations_dir,
        host=agents_db_config["host"],
        port=agents_db_config["port"],
        user=agents_db_config["user"],
        password=agents_db_config["password"],
        database=agents_db_config["db_name"],
        schema=AGENTS_SCHEMA,
    )


@pytest.fixture(scope="session")
def agents_session_maker(migrated_agents_db, agents_db_config):
    return db_manager.get_sessionmaker(**agents_db_config)


@pytest_asyncio.fixture(scope="function")
async def agents_db(agents_session_maker):
    async with agents_session_maker() as session:
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
