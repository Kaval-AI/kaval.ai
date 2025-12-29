import os
from datetime import datetime
from enum import Enum as PyEnum
from uuid import UUID, uuid4

from sqlalchemy import MetaData
from sqlalchemy import TEXT, Boolean, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, ENUM
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.pool import NullPool


def get_database_url():
    return f"postgresql+asyncpg://{os.environ['POSTGRES_DB_USER']}:{os.environ['POSTGRES_DB_PASSWORD']}@{os.environ['POSTGRES_DB_HOST']}:{os.environ['POSTGRES_DB_PORT']}/{os.environ['POSTGRES_DB_NAME']}"


engine = create_async_engine(get_database_url(), echo=True, poolclass=NullPool)

AsyncKavalaiSession = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)


class Base(DeclarativeBase):
    metadata = MetaData(schema=os.environ["POSTGRES_DB_SCHEMA"])


# 5. ORM Models
class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(TEXT, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(TEXT, nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    picture: Mapped[str | None] = mapped_column(TEXT)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow,
                                                 onupdate=datetime.utcnow)

    # Relationships
    memberships: Mapped[list["ProjectMembership"]] = relationship(back_populates="user")


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(TEXT, nullable=False)
    description: Mapped[str | None] = mapped_column(TEXT)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow,
                                                 onupdate=datetime.utcnow)

    # Relationships
    members: Mapped[list["ProjectMembership"]] = relationship(back_populates="project")


class ProjectRole(str, PyEnum):
    owner = "owner"
    viewer = "viewer"


class ProjectMembership(Base):
    __tablename__ = "project_memberships"

    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    project_id: Mapped[UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True)
    role: Mapped[ProjectRole] = mapped_column(
        ENUM(
            ProjectRole,
            name="project_role",
            schema=os.environ.get("POSTGRES_DB_SCHEMA"),  # This is the missing link!
            create_type=False  # Tell SQLAlchemy the type already exists
        ),
        nullable=False
    )

    # Relationship Links
    user: Mapped["User"] = relationship(back_populates="memberships")
    project: Mapped["Project"] = relationship(back_populates="members")
