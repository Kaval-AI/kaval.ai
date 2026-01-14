import os
from datetime import datetime, timezone
from enum import Enum as PyEnum
from uuid import UUID, uuid4

from sqlalchemy import MetaData
from sqlalchemy import TEXT, Boolean, ForeignKey, DateTime, Integer
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, ENUM
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.pool import NullPool


def get_backoffice_db_url():
    return f"postgresql+asyncpg://{os.environ['BACKOFFICE_DB_USER']}:{os.environ['BACKOFFICE_DB_PASSWORD']}@{os.environ['BACKOFFICE_DB_HOST']}:{os.environ['BACKOFFICE_DB_PORT']}/{os.environ['BACKOFFICE_DB_NAME']}"


engine = create_async_engine(get_backoffice_db_url(), echo=False, poolclass=NullPool)

AsyncBackofficeSession = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)


class Base(DeclarativeBase):
    metadata = MetaData(schema=os.environ["BACKOFFICE_DB_SCHEMA"])


class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    email: Mapped[str] = mapped_column(TEXT, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(TEXT, nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    picture: Mapped[str | None] = mapped_column(TEXT)
    active_project_id: Mapped[UUID | None] = mapped_column(ForeignKey("projects.id"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    memberships: Mapped[list["ProjectMembership"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    name: Mapped[str] = mapped_column(TEXT, nullable=False)
    description: Mapped[str | None] = mapped_column(TEXT)

    # New Database Connection Columns
    db_host: Mapped[str | None] = mapped_column(TEXT)
    db_port: Mapped[int | None] = mapped_column(Integer, default=5432)
    db_user: Mapped[str | None] = mapped_column(TEXT)
    db_password: Mapped[str | None] = mapped_column(TEXT)
    db_name: Mapped[str | None] = mapped_column(TEXT)
    db_schema: Mapped[str | None] = mapped_column(TEXT, default="public")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    members: Mapped[list["ProjectMembership"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True
    )


class ProjectRole(str, PyEnum):
    owner = "owner"
    viewer = "viewer"


class ProjectMembership(Base):
    __tablename__ = "project_memberships"

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    role: Mapped[ProjectRole] = mapped_column(
        ENUM(
            ProjectRole,
            name="project_role",
            schema=os.environ.get("BACKOFFICE_DB_SCHEMA"),
            create_type=False,
        ),
        nullable=False,
    )

    # Relationship Links
    user: Mapped["User"] = relationship(back_populates="memberships")
    project: Mapped["Project"] = relationship(back_populates="members")


async def is_member(db: AsyncSession, user_id: UUID, project_id: UUID) -> bool:
    """Check if a user is any kind of member of a project."""
    stmt = select(ProjectMembership).where(
        ProjectMembership.user_id == user_id, ProjectMembership.project_id == project_id
    )
    result = await db.execute(stmt)
    return result.scalars().first() is not None


async def is_owner(db: AsyncSession, user_id: UUID, project_id: UUID) -> bool:
    """Check if a user has the 'owner' role for a project."""
    stmt = select(ProjectMembership).where(
        ProjectMembership.user_id == user_id,
        ProjectMembership.project_id == project_id,
        ProjectMembership.role == ProjectRole.owner,
    )
    result = await db.execute(stmt)
    return result.scalars().first() is not None


async def get_user_projects(db: AsyncSession, user_id: UUID) -> list[dict]:
    """Fetch projects along with the specific user's role and DB details."""
    stmt = (
        select(Project, ProjectMembership.role)
        .join(ProjectMembership, Project.id == ProjectMembership.project_id)
        .where(ProjectMembership.user_id == user_id)
    )

    result = await db.execute(stmt)

    projects_with_roles = []
    for row in result.all():
        project_obj: Project = row[0]
        role: ProjectRole = row[1]

        project_data = {
            "id": str(project_obj.id),
            "name": project_obj.name,
            "description": project_obj.description,
            "db_host": project_obj.db_host,
            "db_port": project_obj.db_port,
            "db_user": project_obj.db_user,
            "db_name": project_obj.db_name,
            "db_schema": project_obj.db_schema,
            "db_password": project_obj.db_password,
            "created_at": project_obj.created_at.isoformat(),
            "updated_at": project_obj.updated_at.isoformat(),
            "role": role.value,
        }
        projects_with_roles.append(project_data)

    return projects_with_roles
