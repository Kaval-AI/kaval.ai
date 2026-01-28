import os
import pytest
import pytest_asyncio
from sqlalchemy import text
from kavalai.migrate_db import migrate
from kavalai import SQL_MIGRATIONS_PATH

BACKOFFICE_SCHEMA = "test_backoffice"
os.environ["BACKOFFICE_DB_SCHEMA"] = BACKOFFICE_SCHEMA


@pytest.fixture(scope="session")
def backoffice_db_config(postgres_container):
    config = dict(
        user=postgres_container.username,
        password=postgres_container.password,
        host=postgres_container.get_container_host_ip(),
        port=int(postgres_container.get_exposed_port(5432)),
        db_name=postgres_container.dbname,
    )
    # Set environment variables BEFORE importing kavalai.backoffice.db
    os.environ["BACKOFFICE_DB_USER"] = config["user"]
    os.environ["BACKOFFICE_DB_PASSWORD"] = config["password"]
    os.environ["BACKOFFICE_DB_HOST"] = config["host"]
    os.environ["BACKOFFICE_DB_PORT"] = str(config["port"])
    os.environ["BACKOFFICE_DB_NAME"] = config["db_name"]
    os.environ["BACKOFFICE_DB_SCHEMA"] = BACKOFFICE_SCHEMA
    return config


@pytest.fixture(scope="session")
def migrated_backoffice_db(backoffice_db_config):
    migrations_dir = os.path.join(SQL_MIGRATIONS_PATH, "backoffice")
    migrate(
        migrations_dir=migrations_dir,
        host=backoffice_db_config["host"],
        port=backoffice_db_config["port"],
        user=backoffice_db_config["user"],
        password=backoffice_db_config["password"],
        database=backoffice_db_config["db_name"],
        schema=BACKOFFICE_SCHEMA,
    )


@pytest_asyncio.fixture(scope="function")
async def backoffice_db(migrated_backoffice_db, backoffice_db_config):
    # Import inside fixture to ensure env vars are set before module-level engine creation
    from kavalai.backoffice.db import AsyncBackofficeSession, Base

    async with AsyncBackofficeSession() as session:
        # Get all table names from the Base metadata to truncate them
        tables = ", ".join(
            f"{BACKOFFICE_SCHEMA}.{table.name}"
            for table in reversed(Base.metadata.sorted_tables)
        )

        if tables:
            await session.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE;"))
            await session.commit()

        yield session
        await session.rollback()
