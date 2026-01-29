import os
import pytest
import psycopg2
from unittest.mock import patch
from testcontainers.postgres import PostgresContainer
from kavalai.migrate_db import migrate, main
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
    uri = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"

    # Run migrations
    migrate(migrations_dir, uri, schema=schema)

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
    schema = "test_backoffice"
    uri = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"

    # Run migrations
    migrate(migrations_dir, uri, schema=schema)

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
    uri = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"

    # First run
    migrate(migrations_dir, uri, schema=schema)

    # Second run should not fail and should not add new entries
    migrate(migrations_dir, uri, schema=schema)

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
    uri = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"

    # Create a temporary migration file
    m_dir = tmp_path / "migrations"
    m_dir.mkdir()
    m1 = m_dir / "V001__test.sql"
    m1.write_text("CREATE TABLE test (id INT);")

    # Apply it
    migrate(str(m_dir), uri, schema=schema)

    # Change the file content
    m1.write_text("CREATE TABLE test (id INT); -- modified")

    # Should raise ValueError
    with pytest.raises(ValueError, match="Checksum mismatch"):
        migrate(str(m_dir), uri, schema=schema)


def test_migrate_main_with_env_vars(db_config):
    schema = "main_env_schema"
    uri = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    env = {
        "KAVALAI_DB_URI": uri,
        "KAVALAI_DB_SCHEMA": schema,
    }

    with patch.dict(os.environ, env), patch("sys.argv", ["migrate_db.py", "app"]):
        main()

    # Verify migrations were applied
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    cur.execute(f"SELECT count(*) FROM {schema}.kavalai_migrations")
    count = cur.fetchone()[0]
    assert count > 0
    conn.close()
