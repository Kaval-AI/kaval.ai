import os
import pytest
import psycopg2
from unittest.mock import patch, MagicMock
from testcontainers.postgres import PostgresContainer
from kavalai.migrate_db import migrate
from kavalai import SQL_MIGRATIONS_PATH


@pytest.fixture(scope="module")
def postgres_container():
    with PostgresContainer("pgvector/pgvector:pg15-trixie") as postgres:
        yield postgres


@pytest.fixture
def db_config(postgres_container):
    return {
        "host": postgres_container.get_container_host_ip(),
        "port": int(postgres_container.get_exposed_port(5432)),
        "user": postgres_container.username,
        "password": postgres_container.password,
        "database": postgres_container.dbname,
    }


def test_migrate_app_migrations(db_config):
    migrations_dir = os.path.join(SQL_MIGRATIONS_PATH, "app")
    schema = "app_schema"

    # Run migrations
    migrate(migrations_dir, schema=schema, **db_config)

    # Verify migrations were applied
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    # Check tracking table
    cur.execute(f"SELECT count(*) FROM {schema}.kavalai_migrations")
    count = cur.fetchone()[0]

    # Count SQL files in app directory
    expected_count = len(
        [
            f
            for f in os.listdir(migrations_dir)
            if f.startswith("V") and f.endswith(".sql")
        ]
    )
    assert count == expected_count

    # Check if one of the tables from migrations exists
    cur.execute(
        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = %s AND table_name = 'agents')",
        (schema,),
    )
    assert cur.fetchone()[0] is True

    conn.close()


def test_migrate_backoffice_migrations(db_config):
    migrations_dir = os.path.join(SQL_MIGRATIONS_PATH, "backoffice")
    schema = "backoffice_schema"

    # Run migrations
    migrate(migrations_dir, schema=schema, **db_config)

    # Verify migrations were applied
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    # Check tracking table
    cur.execute(f"SELECT count(*) FROM {schema}.kavalai_migrations")
    count = cur.fetchone()[0]

    # Count SQL files in backoffice directory
    expected_count = len(
        [
            f
            for f in os.listdir(migrations_dir)
            if f.startswith("V") and f.endswith(".sql")
        ]
    )
    assert count == expected_count

    # Check if one of the tables from migrations exists
    cur.execute(
        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = %s AND table_name = 'users')",
        (schema,),
    )
    assert cur.fetchone()[0] is True

    conn.close()


def test_migrate_idempotency(db_config):
    migrations_dir = os.path.join(SQL_MIGRATIONS_PATH, "app")
    schema = "idempotency_schema"

    # First run
    migrate(migrations_dir, schema=schema, **db_config)

    # Second run should not fail and should not add new entries
    migrate(migrations_dir, schema=schema, **db_config)

    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    cur.execute(f"SELECT count(*) FROM {schema}.kavalai_migrations")
    count = cur.fetchone()[0]

    expected_count = len(
        [
            f
            for f in os.listdir(migrations_dir)
            if f.startswith("V") and f.endswith(".sql")
        ]
    )
    assert count == expected_count
    conn.close()


def test_migrate_checksum_mismatch_real_db(db_config, tmp_path):
    schema = "checksum_schema"

    # Create a temporary migration file
    m_dir = tmp_path / "migrations"
    m_dir.mkdir()
    m1 = m_dir / "V001__test.sql"
    m1.write_text("CREATE TABLE test (id INT);")

    # Apply it
    migrate(str(m_dir), schema=schema, **db_config)

    # Change the file content
    m1.write_text("CREATE TABLE test (id INT); -- modified")

    # Should raise ValueError
    with pytest.raises(ValueError, match="Checksum mismatch"):
        migrate(str(m_dir), schema=schema, **db_config)


def test_migrate_retries_on_connection_failure():
    """
    Test that migrate function retries connection on failure.
    We mock psycopg2.connect to fail a few times and then succeed.
    """
    with patch("psycopg2.connect") as mock_connect:
        # Mock failure then success
        mock_connect.side_effect = [
            psycopg2.OperationalError("Connection refused"),
            psycopg2.OperationalError("Connection refused"),
            MagicMock(),  # Success on 3rd attempt
        ]

        # We also need to mock other things that migrate calls after successful connection
        with patch("kavalai.migrate_db.ensure_schema_and_table"), patch(
            "kavalai.migrate_db.get_applied_migrations", return_value={}
        ), patch("kavalai.migrate_db.get_migrations", return_value=[]), patch(
            "time.sleep"
        ) as mock_sleep:  # Mock sleep to speed up test
            migrate(
                migrations_dir="fake_dir",
                host="localhost",
                port=5432,
                user="user",
                password="password",
                database="db",
                schema="public",
            )

            assert mock_connect.call_count == 3
            assert mock_sleep.call_count == 2
