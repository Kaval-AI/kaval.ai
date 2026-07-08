"""
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Shared logic for the per-set Alembic ``env.py`` files.

The runner (:mod:`kavalai.migrate_db`) passes the database connection and the
target schema in via ``config.attributes`` — the env files never read
environment variables. The schema is applied with ``schema_translate_map``, so
migration scripts stay schema-less (matching the schema-less ORM metadata) and
the ``alembic_version`` table lands in the target schema alongside the tables.
"""

from alembic import context
from sqlalchemy.engine import Connection


def agents_include_object(obj, name, type_, reflected, compare_to):
    """Keep autogenerate away from backend-owned objects (agents set).

    All ``rag_*`` tables belong to the self-provisioning RAG backends
    (``rag_collections`` registry + one table per collection) and are never
    part of this migration set. Shared with the parity tests.
    """
    if type_ == "table" and name.startswith("rag_"):
        return False
    return True


def agents_render_item(type_, obj, autogen_context):
    """Render kavalai's dialect-variant column types with proper imports.

    Without this, autogenerate renders the Postgres base type (``JSONB``,
    ``UUID``) and drops the SQLite variant, breaking revisions on SQLite.
    """
    if type_ != "type":
        return False

    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.dialects.postgresql import UUID as PG_UUID

    from kavalai.agents.db import VectorType

    imports = "from kavalai.agents.db import VectorType, json_column, uuid_column"
    if isinstance(obj, VectorType):
        autogen_context.imports.add(imports)
        return "VectorType()"
    if isinstance(obj, PG_UUID):
        autogen_context.imports.add(imports)
        return "uuid_column()"
    if isinstance(obj, JSONB):
        autogen_context.imports.add(imports)
        return "json_column()"
    return False


def run_migrations(target_metadata, render_item=None, include_object=None):
    """Run Alembic migrations for one migration set.

    Connection resolution order:
    1. ``config.attributes["connection"]`` — a live ``Connection`` provided by
       the runner (transaction managed by the caller).
    2. ``sqlalchemy.url`` from the Alembic config — used by dev CLI invocations
       (e.g. autogenerate); the target schema may be passed as
       ``-x schema=<name>``.
    """
    config = context.config

    schema = config.attributes.get("schema")
    if schema is None:
        schema = context.get_x_argument(as_dictionary=True).get("schema")

    connectable = config.attributes.get("connection")
    if isinstance(connectable, Connection):
        _configure_and_run(
            connectable, target_metadata, schema, render_item, include_object
        )
        return

    from sqlalchemy import engine_from_config, pool

    engine = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    try:
        with engine.connect() as connection:
            _configure_and_run(
                connection, target_metadata, schema, render_item, include_object
            )
            connection.commit()
    finally:
        engine.dispose()


def _configure_and_run(
    connection, target_metadata, schema, render_item, include_object
):
    if schema:
        connection = connection.execution_options(schema_translate_map={None: schema})
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # The translate map covers compiled DDL/DML, but Alembic's version
        # table checks reflect with an explicit schema — give it the real one.
        version_table_schema=schema,
        # SQLite cannot ALTER TABLE in place; batch mode rebuilds the table.
        render_as_batch=True,
        compare_type=True,
        render_item=render_item,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()
