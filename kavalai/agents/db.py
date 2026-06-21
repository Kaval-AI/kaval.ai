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
"""

from datetime import datetime, timezone
from uuid import UUID, uuid4

from environs import Env
from sqlalchemy import (
    JSON,
    MetaData,
    TEXT,
    ForeignKey,
    DateTime,
    Integer,
    Numeric,
    TypeDecorator,
    Uuid,
    create_engine,
    event,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)
from sqlalchemy.pool import StaticPool

from kavalai.agents import idb


def uuid_column():
    """UUID column type.

    Uses PostgreSQL's native ``uuid`` on Postgres and SQLAlchemy's generic
    :class:`~sqlalchemy.Uuid` on SQLite, so the same ORM models work against
    both the production Postgres backend and the SQLite/IndexedDB backend used
    under Pyodide.
    """
    return PG_UUID(as_uuid=True).with_variant(Uuid(as_uuid=True), "sqlite")


def json_column():
    """JSON column type: ``JSONB`` on Postgres, generic ``JSON`` on SQLite."""
    return JSONB().with_variant(JSON(), "sqlite")


def parse_db_uri(uri: str) -> dict:
    """Parses a database URI into tokens."""
    url = make_url(uri)
    return {
        "user": url.username,
        "password": url.password,
        "host": url.host,
        "port": url.port,
        "db_name": url.database,
    }


class VectorType(TypeDecorator):
    impl = TEXT
    cache_ok = True

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect):
        # Outside PostgreSQL (e.g. the SQLite/IndexedDB backend used under
        # Pyodide) the ``pgvector`` type is unavailable, so fall back to the
        # TEXT representation produced by ``process_bind_param``.
        if dialect.name != "postgresql":
            return dialect.type_descriptor(TEXT())

        from sqlalchemy.types import UserDefinedType

        class Vector(UserDefinedType):
            def __init__(self, dim=None):
                self.dim = dim

            def get_col_spec(self, **kw):
                if self.dim:
                    return f"public.vector({self.dim})"
                return "public.vector"

        return dialect.type_descriptor(Vector(self.dim))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return str(list(value))

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        # value is already a list or string depending on driver
        if isinstance(value, str):
            return [float(x) for x in value.strip("[]").split(",")]
        return value


def ensure_async_scheme(uri: str) -> str:
    """Ensures the URI uses the postgresql+asyncpg driver."""
    if uri and "://" in uri:
        scheme, rest = uri.split("://", 1)
        if scheme.startswith("postgresql"):
            return f"postgresql+asyncpg://{rest}"
    return uri


def build_db_uri(
    user, password, host, port, db_name, scheme="postgresql+asyncpg"
) -> str:
    """Builds a database URI from components."""
    return f"{scheme}://{user}:{password}@{host}:{port}/{db_name}"


class DatabaseManager:
    """Manages dynamic engine creation and session factories."""

    # Default location of the SQLite file when running under Pyodide. It lives
    # inside the IDBFS mount point so it is persisted to the browser's
    # IndexedDB (see :mod:`kavalai.agents.idb`).
    SQLITE_PYODIDE_DB_PATH = f"{idb.MOUNT_DIR}/kavalai.db"

    def __init__(self):
        self._engines = {}

    def get_sessionmaker(
        self,
        *,
        user=None,
        password=None,
        host=None,
        port=None,
        db_name=None,
        uri=None,
        echo=False,
        pool_size=1,
        max_overflow=0,
    ):
        if uri:
            url = ensure_async_scheme(uri)
        else:
            url = build_db_uri(user, password, host, port, db_name)

        # Check cache first
        if url not in self._engines:
            self._engines[url] = create_async_engine(
                url,
                echo=echo,
                pool_size=pool_size,
                max_overflow=max_overflow,
            )
        engine = self._engines[url]
        return async_sessionmaker(
            bind=engine, class_=AsyncSession, expire_on_commit=False
        )

    # -- SQLite / IndexedDB backend -------------------------------------------
    #
    # A pure-Python, pyodide-compatible backend. The ORM models live in a named
    # schema (``KAVALAI_DB_SCHEMA``, default ``"test"``); SQLite exposes that
    # schema by ATTACH-ing the database file under that alias. Under Pyodide the
    # file sits on IDBFS so it is persisted to the browser's IndexedDB.
    #
    # Two flavours are offered: an async engine (for normal CPython, where
    # ``greenlet`` is available) and a *sync* engine. ``greenlet`` has no pyodide
    # build, and SQLAlchemy's async engine depends on it, so in the browser the
    # sync engine is the one that works -- see :meth:`get_sqlite_sync_engine`.

    def _resolve_sqlite_target(self, db_path, schema):
        """Resolve the ATTACH alias (schema) and database file path."""
        # The ATTACH alias must match the schema the ORM tables actually live in
        # (fixed on ``Base.metadata`` at import time), otherwise schema-qualified
        # DDL like ``CREATE TABLE <schema>.agents`` targets an unknown database.
        schema = schema or Base.metadata.schema
        if db_path is None:
            db_path = self.SQLITE_PYODIDE_DB_PATH if idb.is_pyodide() else ":memory:"
        return db_path, schema

    @staticmethod
    def _attach_listener(db_path, schema):
        """Build a connect listener that enables FKs and ATTACHes the schema db."""
        attach_sql = f'ATTACH DATABASE "{db_path}" AS "{schema}"'

        def _configure_connection(dbapi_connection, _record):
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute(attach_sql)
            finally:
                cursor.close()

        return _configure_connection

    def get_sqlite_engine(self, *, db_path=None, schema=None, echo=False):
        """Return a cached **async** SQLite engine backed by ``db_path``.

        Requires ``greenlet`` (SQLAlchemy's async engine dependency), so this is
        for CPython. Under pyodide use :meth:`get_sqlite_sync_engine` instead.

        ``db_path`` defaults to the IndexedDB-backed file under Pyodide and to an
        in-memory database otherwise. The configured schema is exposed by
        ATTACH-ing ``db_path`` under that name, and SQLite foreign keys are
        enabled. A :class:`~sqlalchemy.pool.StaticPool` keeps a single shared
        connection so the in-memory main database and the ATTACH survive across
        sessions.
        """
        db_path, schema = self._resolve_sqlite_target(db_path, schema)
        key = f"sqlite-async::{db_path}::{schema}"
        if key not in self._engines:
            engine = create_async_engine(
                "sqlite+aiosqlite://", echo=echo, poolclass=StaticPool
            )
            event.listen(
                engine.sync_engine, "connect", self._attach_listener(db_path, schema)
            )
            self._engines[key] = engine
        return self._engines[key]

    def get_sqlite_sessionmaker(self, *, db_path=None, schema=None, echo=False):
        """Return an ``async_sessionmaker`` bound to the async SQLite engine."""
        engine = self.get_sqlite_engine(db_path=db_path, schema=schema, echo=echo)
        return async_sessionmaker(
            bind=engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_sqlite(self, *, db_path=None, schema=None, echo=False):
        """Prepare the async SQLite/IndexedDB backend (mount, create tables, persist)."""
        await idb.mount()
        engine = self.get_sqlite_engine(db_path=db_path, schema=schema, echo=echo)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await idb.flush()

    def get_sqlite_sync_engine(self, *, db_path=None, schema=None, echo=False):
        """Return a cached **sync** SQLite engine backed by ``db_path``.

        This is the pyodide-friendly variant: SQLAlchemy's synchronous engine
        does not need ``greenlet`` (which has no pyodide build). It behaves like
        :meth:`get_sqlite_engine` otherwise -- ATTACH-ing ``db_path`` under the
        configured schema with foreign keys enabled and a shared connection.
        """
        db_path, schema = self._resolve_sqlite_target(db_path, schema)
        key = f"sqlite-sync::{db_path}::{schema}"
        if key not in self._engines:
            engine = create_engine("sqlite://", echo=echo, poolclass=StaticPool)
            event.listen(engine, "connect", self._attach_listener(db_path, schema))
            self._engines[key] = engine
        return self._engines[key]

    def get_sqlite_sync_sessionmaker(self, *, db_path=None, schema=None, echo=False):
        """Return a sync ``sessionmaker`` bound to the sync SQLite engine."""
        engine = self.get_sqlite_sync_engine(db_path=db_path, schema=schema, echo=echo)
        return sessionmaker(bind=engine, expire_on_commit=False)

    async def init_sqlite_sync(self, *, db_path=None, schema=None, echo=False):
        """Prepare the sync SQLite/IndexedDB backend (mount, create tables, persist).

        Mounts IndexedDB (under Pyodide), creates all ORM tables and persists the
        resulting database. The table creation itself is synchronous; only the
        IndexedDB mount/flush are awaited. Safe to call repeatedly.
        """
        await idb.mount()
        engine = self.get_sqlite_sync_engine(db_path=db_path, schema=schema, echo=echo)
        Base.metadata.create_all(engine)
        await idb.flush()

    async def flush(self):
        """Persist pending SQLite changes to IndexedDB (no-op outside Pyodide)."""
        await idb.flush()


# Global instance to manage cached engines
db_manager = DatabaseManager()


def get_kavalai_db_schema():
    env = Env()
    env.read_env()
    return env.str("KAVALAI_DB_SCHEMA", "test")


class Base(DeclarativeBase):
    metadata = MetaData(schema=get_kavalai_db_schema())


class Agent(Base):
    """A configured agent.

    One row holds an agent's definition: its name, optional description, the
    JSON input/output schemas it exposes, and the workflow document that drives
    its execution. Sessions, tasks and chat messages all reference the agent
    they belong to.
    """

    __tablename__ = "agents"

    id: Mapped[UUID] = mapped_column(uuid_column(), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(TEXT, nullable=False)
    description: Mapped[str | None] = mapped_column(TEXT)
    input_schema: Mapped[dict | None] = mapped_column(json_column())
    output_schema: Mapped[dict | None] = mapped_column(json_column())
    workflow: Mapped[dict | None] = mapped_column(
        json_column()
    )  # Updated to match SQL 'workflow'
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    sessions: Mapped[list["Session"]] = relationship(
        back_populates="agent", cascade="all, delete-orphan", passive_deletes=True
    )
    tasks: Mapped[list["Task"]] = relationship(back_populates="agent")
    chat_messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="agent", cascade="all, delete-orphan", passive_deletes=True
    )


class ModelCallStat(Base):
    """Token usage and timing for a single LLM or embedding call.

    Each row records one model call (``call_type`` distinguishes e.g. chat
    completion from embedding): the model used, the request/response payloads,
    prompt/completion/total token counts, batch size, duration and the
    computed cost. Used to power usage and cost reporting.
    """

    __tablename__ = "model_call_stats"

    id: Mapped[UUID] = mapped_column(uuid_column(), primary_key=True, default=uuid4)
    call_type: Mapped[str] = mapped_column(TEXT, nullable=False)
    model: Mapped[str] = mapped_column(TEXT, nullable=False)
    agent_id: Mapped[UUID | None] = mapped_column(uuid_column())
    request_data: Mapped[dict | None] = mapped_column(json_column())
    response_data: Mapped[dict | None] = mapped_column(json_column())
    response_code: Mapped[int | None] = mapped_column(Integer)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    total_tokens: Mapped[int | None] = mapped_column(Integer)
    batch_size: Mapped[int | None] = mapped_column(Integer)
    duration_seconds: Mapped[float | None] = mapped_column(Numeric(10, 6))
    cost: Mapped[float | None] = mapped_column(Numeric(10, 6))
    currency: Mapped[str | None] = mapped_column(TEXT)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class Session(Base):
    """A user conversation/session with an agent.

    A session groups together everything exchanged with one agent over a
    conversation: its runs, tasks and chat messages. ``external_id`` lets a
    caller correlate the session with an identifier in their own system.
    """

    __tablename__ = "sessions"

    id: Mapped[UUID] = mapped_column(uuid_column(), primary_key=True, default=uuid4)
    agent_id: Mapped[UUID] = mapped_column(
        ForeignKey("agents.id", ondelete="CASCADE"), nullable=False
    )
    external_id: Mapped[str | None] = mapped_column(TEXT)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    agent: Mapped["Agent"] = relationship(back_populates="sessions")
    runs: Mapped[list["Run"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", passive_deletes=True
    )
    tasks: Mapped[list["Task"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", passive_deletes=True
    )
    chat_messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", passive_deletes=True
    )


class Run(Base):
    """One workflow run within a session.

    A run captures a single invocation of the agent's workflow: the input it
    was called with, the output it produced and the resolved run context. Each
    run belongs to a session and owns the tasks executed during it.
    """

    __tablename__ = "runs"  # Renamed from 'interactions'

    id: Mapped[UUID] = mapped_column(uuid_column(), primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    input_data: Mapped[dict | None] = mapped_column(json_column())
    output_data: Mapped[dict | None] = mapped_column(json_column())
    context: Mapped[dict | None] = mapped_column(json_column())
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    session: Mapped["Session"] = relationship(back_populates="runs")
    tasks: Mapped[list["Task"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )
    chat_messages: Mapped[list["ChatMessage"]] = relationship(back_populates="run")


class Task(Base):
    """One workflow-node execution within a run.

    A task records the execution of a single node in the workflow: its name and
    node type, the inputs it received, the output it produced, the prompt used
    (if any), any errors raised and how long it took. Tasks belong to a run (and
    its session) and provide the step-by-step trace of a run.
    """

    __tablename__ = "tasks"

    id: Mapped[UUID] = mapped_column(uuid_column(), primary_key=True, default=uuid4)
    agent_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("agents.id", ondelete="SET NULL")
    )
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    run_id: Mapped[UUID] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    inputs: Mapped[dict | None] = mapped_column(json_column())
    output: Mapped[dict | None] = mapped_column(json_column())
    name: Mapped[str | None] = mapped_column(TEXT)
    node_type: Mapped[str | None] = mapped_column(TEXT)
    prompt: Mapped[str | None] = mapped_column(TEXT)
    errors: Mapped[list[str] | None] = mapped_column(json_column())
    duration_seconds: Mapped[float | None] = mapped_column(Numeric)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    agent: Mapped["Agent"] = relationship(back_populates="tasks")
    session: Mapped["Session"] = relationship(back_populates="tasks")
    run: Mapped["Run"] = relationship(back_populates="tasks")


class ChatMessage(Base):
    """One message in a session's chat history.

    Each row is a single message (``role`` such as ``user`` or ``assistant``
    with its text ``content``) belonging to a session. It optionally references
    the run that produced it, forming the ordered conversation transcript.
    """

    __tablename__ = "chat_messages"

    id: Mapped[UUID] = mapped_column(uuid_column(), primary_key=True, default=uuid4)
    agent_id: Mapped[UUID] = mapped_column(
        ForeignKey("agents.id", ondelete="CASCADE"), nullable=False
    )
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    run_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("runs.id", ondelete="SET NULL")
    )
    role: Mapped[str] = mapped_column(TEXT, nullable=False)
    content: Mapped[str] = mapped_column(TEXT, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    agent: Mapped["Agent"] = relationship(back_populates="chat_messages")
    session: Mapped["Session"] = relationship(back_populates="chat_messages")
    run: Mapped["Run"] = relationship(back_populates="chat_messages")


class RagIndex(Base):
    """One indexed RAG item and its embedding.

    Each row stores a chunk of content together with the embedding vector used
    for similarity search: the embedding ``model`` and vector size, the
    ``collection_name`` it belongs to, a caller-supplied ``source_id`` and any
    associated metadata. Queried for retrieval-augmented generation.
    """

    __tablename__ = "rag_index"

    id: Mapped[UUID] = mapped_column(uuid_column(), primary_key=True, default=uuid4)
    model: Mapped[str] = mapped_column(TEXT, nullable=False)
    collection_name: Mapped[str] = mapped_column(TEXT, nullable=False)
    source_id: Mapped[str] = mapped_column(TEXT, nullable=False)
    content: Mapped[str | None] = mapped_column(TEXT)
    embedding_size: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(VectorType())
    rag_metadata: Mapped[dict | None] = mapped_column("metadata", json_column())

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
