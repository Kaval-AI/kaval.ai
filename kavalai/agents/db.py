import os
from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import MetaData, TEXT, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.pool import NullPool


def get_agents_db_url():
    return f"postgresql+asyncpg://{os.environ['AGENTS_DB_USER']}:{os.environ['AGENTS_DB_PASSWORD']}@{os.environ['AGENTS_DB_HOST']}:{os.environ['AGENTS_DB_PORT']}/{os.environ['AGENTS_DB_NAME']}"


engine = create_async_engine(get_agents_db_url(), echo=True, poolclass=NullPool)

AsyncAgentsSession = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)


class Base(DeclarativeBase):
    # Set schema from env, similar to your backoffice setup
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
    template_variables: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    sessions: Mapped[list["Session"]] = relationship(
        back_populates="agent", cascade="all, delete-orphan", passive_deletes=True
    )
    chat_messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="agent", cascade="all, delete-orphan", passive_deletes=True
    )


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

    # Relationships
    agent: Mapped["Agent"] = relationship(back_populates="sessions")
    interactions: Mapped[list["Interaction"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", passive_deletes=True
    )
    chat_messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", passive_deletes=True
    )


class Interaction(Base):
    __tablename__ = "interactions"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    input_data: Mapped[dict | None] = mapped_column(JSONB)
    output_data: Mapped[dict | None] = mapped_column(JSONB)
    steps: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    session: Mapped["Session"] = relationship(back_populates="interactions")
    chat_messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="interaction"
    )


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
    interaction_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("interactions.id", ondelete="SET NULL")
    )
    role: Mapped[str] = mapped_column(
        TEXT, nullable=False
    )  # e.g., 'user', 'assistant', 'system'
    content: Mapped[str] = mapped_column(TEXT, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    agent: Mapped["Agent"] = relationship(back_populates="chat_messages")
    session: Mapped["Session"] = relationship(back_populates="chat_messages")
    interaction: Mapped["Interaction"] = relationship(back_populates="chat_messages")
