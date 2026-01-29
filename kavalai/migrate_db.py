import argparse
import hashlib
import logging
import os
import time
from typing import List, Tuple

import psycopg2
from psycopg2 import sql

from kavalai import SQL_MIGRATIONS_PATH

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def calculate_checksum(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_migrations(migrations_dir: str) -> List[Tuple[str, str]]:
    migrations = []
    if not os.path.exists(migrations_dir):
        logger.error(f"Migrations directory not found: {migrations_dir}")
        return []

    for filename in os.listdir(migrations_dir):
        if filename.endswith(".sql") and filename.startswith("V"):
            migrations.append(filename)

    # Sort by version (V001, V002, etc.)
    migrations.sort()

    return [
        (filename, os.path.join(migrations_dir, filename)) for filename in migrations
    ]


def ensure_schema_and_table(cur, schema: str):
    """Ensures the schema and the migration tracking table exist."""
    cur.execute(
        sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
    )
    # Set search_path to the target schema and public (for extensions)
    cur.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))

    # Create migrations table if it doesn't exist
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {schema}.kavalai_migrations (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255) UNIQUE NOT NULL,
            checksum VARCHAR(64) NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


def get_applied_migrations(cur, schema: str) -> dict:
    """Fetches the already applied migrations from the tracking table."""
    cur.execute(f"SELECT filename, checksum FROM {schema}.kavalai_migrations")
    return {row[0]: row[1] for row in cur.fetchall()}


def apply_migration(cur, schema: str, filename: str, file_path: str, checksum: str):
    """Reads and executes a single migration file and records it in the tracking table."""
    logger.info(f"Applying migration: {filename}")
    with open(file_path, "r") as f:
        sql_content = f.read()

    cur.execute(sql_content)
    cur.execute(
        f"INSERT INTO {schema}.kavalai_migrations (filename, checksum) VALUES (%s, %s)",
        (filename, checksum),
    )


def migrate(
    migrations_dir: str,
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    schema: str,
):
    conn = None
    max_wait = 60
    start_time = time.time()
    retry_interval = 1

    while True:
        try:
            conn = psycopg2.connect(
                host=host, port=port, user=user, password=password, dbname=database
            )
            break
        except psycopg2.OperationalError:
            elapsed = time.time() - start_time
            if elapsed > max_wait:
                logger.error(f"Failed to connect to database after {max_wait} seconds.")
                raise
            logger.info(
                f"Database connection failed, retrying in {retry_interval}s... ({int(elapsed)}s elapsed)"
            )
            time.sleep(retry_interval)
            retry_interval = min(
                retry_interval * 2, max_wait - (time.time() - start_time)
            )
            if retry_interval <= 0:
                retry_interval = 0.1

    try:
        conn.autocommit = False
        cur = conn.cursor()

        ensure_schema_and_table(cur, schema)
        applied_migrations = get_applied_migrations(cur, schema)

        migrations = get_migrations(migrations_dir)
        if not migrations:
            logger.info("No migrations found.")
            conn.commit()
            return

        for filename, file_path in migrations:
            checksum = calculate_checksum(file_path)

            if filename in applied_migrations:
                if applied_migrations[filename] != checksum:
                    logger.error(
                        f"Checksum mismatch for migration {filename}. Expected {applied_migrations[filename]}, got {checksum}."
                    )
                    conn.rollback()
                    raise ValueError(f"Checksum mismatch for {filename}")
                logger.info(f"Migration {filename} already applied.")
                continue

            try:
                apply_migration(cur, schema, filename, file_path, checksum)
            except Exception as e:
                logger.error(f"Error applying migration {filename}: {e}")
                conn.rollback()
                raise

        conn.commit()
        logger.info("Migrations completed successfully.")

    except Exception as e:
        logger.error(f"Database error: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def main():
    parser = argparse.ArgumentParser(description="Kaval.AI Database Migration Tool")
    parser.add_argument(
        "type", choices=["app", "backoffice"], help="Type of migrations to run."
    )
    args = parser.parse_args()

    if args.type == "backoffice":
        host = os.environ.get("BACKOFFICE_DB_HOST")
        port = int(os.environ.get("BACKOFFICE_DB_PORT", 5432))
        user = os.environ.get("BACKOFFICE_DB_USER")
        password = os.environ.get("BACKOFFICE_DB_PASSWORD")
        database = os.environ.get("BACKOFFICE_DB_NAME")
        schema = os.environ.get("BACKOFFICE_DB_SCHEMA")
        migrations_dir = os.path.join(SQL_MIGRATIONS_PATH, "backoffice")
    elif args.type == "app":
        host = os.environ.get("AGENTS_DB_HOST")
        port = int(os.environ.get("AGENTS_DB_PORT", 5432))
        user = os.environ.get("AGENTS_DB_USER")
        password = os.environ.get("AGENTS_DB_PASSWORD")
        database = os.environ.get("AGENTS_DB_NAME")
        schema = os.environ.get("AGENTS_DB_SCHEMA")
        migrations_dir = os.path.join(SQL_MIGRATIONS_PATH, "app")
    else:
        raise ValueError(f"Invalid migration type: {args.type}")

    migrate(
        migrations_dir,
        host,
        port,
        user,
        password,
        database,
        schema,
    )


if __name__ == "__main__":
    main()
