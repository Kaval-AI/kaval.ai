import os
import pytest
import pytest_asyncio
from sqlalchemy import text
from kavalai.migrate_db import migrate
from kavalai.db import build_db_uri


# The backoffice application reads these at call time (AsyncBackofficeSession
# defaults); the library itself never reads them at import time anymore.
os.environ["KAVALAI_BO_DB_SCHEMA"] = "test_backoffice"


@pytest.fixture(scope="session")
def backoffice_db_config(postgres_container):
    config = dict(
        uri=build_db_uri(
            user=postgres_container.username,
            password=postgres_container.password,
            host=postgres_container.get_container_host_ip(),
            port=int(postgres_container.get_exposed_port(5432)),
            db_name=postgres_container.dbname,
        ),
        schema="test_backoffice",
    )
    # Set environment variables BEFORE importing kavalai.backoffice.db
    os.environ["KAVALAI_BO_DB_URI"] = config["uri"]
    os.environ["KAVALAI_BO_DB_SCHEMA"] = "test_backoffice"
    return config


@pytest.fixture(scope="session")
def migrated_backoffice_db(backoffice_db_config):
    migrate(
        "backoffice",
        uri=backoffice_db_config["uri"],
        schema=backoffice_db_config["schema"],
    )


@pytest_asyncio.fixture(scope="function")
async def backoffice_db(migrated_backoffice_db, backoffice_db_config):
    # Import inside fixture to ensure env vars are set before module-level engine creation
    from kavalai.backoffice.db import AsyncBackofficeSession, Base

    async with AsyncBackofficeSession() as session:
        # Get all table names from the Base metadata to truncate them
        tables = ", ".join(
            f"test_backoffice.{table.name}"
            for table in reversed(Base.metadata.sorted_tables)
        )

        if tables:
            await session.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE;"))
            await session.commit()

        yield session
        await session.rollback()
