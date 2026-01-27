import os
from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import (
    MetaData,
    TEXT,
    ForeignKey,
    DateTime,
    Integer,
    Numeric,
    TypeDecorator,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.pool import NullPool


class VectorType(TypeDecorator):
    impl = TEXT
    cache_ok = True

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect):
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


class DatabaseManager:
    """Manages dynamic engine creation and session factories."""

    def __init__(self):
        # Optional: Cache engines by a unique key (e.g., host+db_name)
        # to avoid the overhead of re-creating engines constantly.
        self._engines = {}

    def get_url(self, user, password, host, port, db_name) -> str:
        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db_name}"

    def get_sessionmaker(self, *, user, password, host, port, db_name, echo=False):
        url = self.get_url(user, password, host, port, db_name)

        # Check cache first
        if url not in self._engines:
            self._engines[url] = create_async_engine(url, echo=echo, poolclass=NullPool)

        engine = self._engines[url]
        return async_sessionmaker(
            bind=engine, class_=AsyncSession, expire_on_commit=False
        )


# Global instance to manage cached engines
db_manager = DatabaseManager()


class Base(DeclarativeBase):
    metadata = MetaData(schema=os.environ.get("AGENTS_DB_SCHEMA"))


class Agent(Base):
    __tablename__ = "agents"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    name: Mapped[str] = mapped_column(TEXT, nullable=False)
    description: Mapped[str | None] = mapped_column(TEXT)
    input_schema: Mapped[dict | None] = mapped_column(JSONB)
    output_schema: Mapped[dict | None] = mapped_column(JSONB)
    workflow: Mapped[dict | None] = mapped_column(
        JSONB
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


class LLMProfile(Base):
    __tablename__ = "llm_profiles"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    name: Mapped[str] = mapped_column(TEXT, nullable=False)
    provider: Mapped[str] = mapped_column(TEXT, nullable=False)
    model_name: Mapped[str] = mapped_column(TEXT, nullable=False)
    api_key: Mapped[str | None] = mapped_column(TEXT)
    base_url: Mapped[str | None] = mapped_column(TEXT)
    config: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    call_stats: Mapped[list["LLMCallStat"]] = relationship(
        back_populates="llm_profile", cascade="all, delete-orphan", passive_deletes=True
    )


class LLMCallStat(Base):
    __tablename__ = "llm_call_stats"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    llm_profile_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("llm_profiles.id", ondelete="SET NULL")
    )
    agent_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("agents.id", ondelete="SET NULL")
    )
    request_data: Mapped[dict | None] = mapped_column(JSONB)
    response_data: Mapped[dict | None] = mapped_column(JSONB)
    response_code: Mapped[int | None] = mapped_column(Integer)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    total_tokens: Mapped[int | None] = mapped_column(Integer)
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

    llm_profile: Mapped["LLMProfile"] = relationship(back_populates="call_stats")
    agent: Mapped["Agent"] = relationship()


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
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
    __tablename__ = "runs"  # Renamed from 'interactions'

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    input_data: Mapped[dict | None] = mapped_column(JSONB)
    output_data: Mapped[dict | None] = mapped_column(JSONB)
    context: Mapped[dict | None] = mapped_column(JSONB)
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
    __tablename__ = "tasks"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    agent_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("agents.id", ondelete="SET NULL")
    )
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    run_id: Mapped[UUID] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    inputs: Mapped[dict | None] = mapped_column(JSONB)
    output: Mapped[dict | None] = mapped_column(JSONB)
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
    __tablename__ = "chat_messages"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
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


class EmbeddingProfile(Base):
    __tablename__ = "embedding_profiles"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    name: Mapped[str] = mapped_column(TEXT, nullable=False)
    provider: Mapped[str] = mapped_column(TEXT, nullable=False)
    model_name: Mapped[str] = mapped_column(TEXT, nullable=False)
    api_key: Mapped[str | None] = mapped_column(TEXT)
    base_url: Mapped[str | None] = mapped_column(TEXT)
    embedding_size: Mapped[int | None] = mapped_column(Integer)
    config: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    rag_items: Mapped[list["RagIndex"]] = relationship(
        back_populates="embedding_profile",
        cascade="all, delete-orphan",
        passive_deletes=False,
    )


class EmbeddingCallStat(Base):
    __tablename__ = "embedding_call_stats"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    embedding_profile_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("embedding_profiles.id", ondelete="SET NULL")
    )
    agent_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("agents.id", ondelete="SET NULL")
    )
    request_data: Mapped[dict | None] = mapped_column(JSONB)
    response_data: Mapped[dict | None] = mapped_column(JSONB)
    response_code: Mapped[int | None] = mapped_column(Integer)
    batch_size: Mapped[int | None] = mapped_column(Integer)
    total_tokens: Mapped[int | None] = mapped_column(Integer)
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

    embedding_profile: Mapped["EmbeddingProfile"] = relationship()
    agent: Mapped["Agent"] = relationship()


class RagIndex(Base):
    __tablename__ = "rag_index"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    embedding_profile_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("embedding_profiles.id", ondelete="CASCADE")
    )
    collection_name: Mapped[str] = mapped_column(TEXT, nullable=False)
    source_id: Mapped[str] = mapped_column(TEXT, nullable=False)
    content: Mapped[str | None] = mapped_column(TEXT)
    embedding_size: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(VectorType())
    rag_metadata: Mapped[dict | None] = mapped_column("metadata", JSONB)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    embedding_profile: Mapped["EmbeddingProfile"] = relationship(
        back_populates="rag_items"
    )
