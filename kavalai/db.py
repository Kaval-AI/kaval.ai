import os
from datetime import datetime, timezone
from enum import Enum as PyEnum
from typing import Type, TypeVar, Sequence, Any
from uuid import UUID, uuid4

from sqlalchemy import MetaData
from sqlalchemy import TEXT, Boolean, ForeignKey, DateTime
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, ENUM
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.pool import NullPool


def get_database_url():
    return f"postgresql+asyncpg://{os.environ['POSTGRES_DB_USER']}:{os.environ['POSTGRES_DB_PASSWORD']}@{os.environ['POSTGRES_DB_HOST']}:{os.environ['POSTGRES_DB_PORT']}/{os.environ['POSTGRES_DB_NAME']}"


engine = create_async_engine(get_database_url(), echo=True, poolclass=NullPool)

AsyncKavalaiSession = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)


class Base(DeclarativeBase):
    metadata = MetaData(schema=os.environ["POSTGRES_DB_SCHEMA"])


class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    email: Mapped[str] = mapped_column(TEXT, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(TEXT, nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    picture: Mapped[str | None] = mapped_column(TEXT)
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
            schema=os.environ.get("POSTGRES_DB_SCHEMA"),
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
    """Fetch projects along with the specific user's role."""
    stmt = (
        select(Project, ProjectMembership.role)
        .join(ProjectMembership, Project.id == ProjectMembership.project_id)
        .where(ProjectMembership.user_id == user_id)
    )

    result = await db.execute(stmt)
    # result contains Rows of (Project, ProjectRole)

    projects_with_roles = []
    for row in result.all():
        project_obj: Project = row[0]
        role: ProjectRole = row[1]

        # Flatten the data to match your TypeScript 'ProjectWithRole' interface
        project_data = {
            "id": str(project_obj.id),
            "name": project_obj.name,
            "description": project_obj.description,
            "created_at": project_obj.created_at.isoformat(),
            "updated_at": project_obj.updated_at.isoformat(),
            "role": role.value,  # The string value "owner" or "viewer"
        }
        projects_with_roles.append(project_data)

    return projects_with_roles


# Type variable to represent any SQLAlchemy model
T = TypeVar("T", bound=Base)


async def get_all(db: AsyncSession, model: Type[T]) -> Sequence[T]:
    """Fetch all records for the model."""
    result = await db.execute(select(model))
    return result.scalars().all()


async def get_one(db: AsyncSession, model: Type[T], id: Any) -> T | None:
    """Fetch a single record by its primary key."""
    return await db.get(model, id)


async def insert(db: AsyncSession, model: Type[T], data: dict) -> T:
    """Create a new record."""
    instance = model(**data)
    db.add(instance)
    await db.commit()
    await db.refresh(instance)
    return instance


async def update(db: AsyncSession, model: Type[T], id: Any, data: dict) -> T | None:
    """Update an existing record by ID."""
    instance = await get_one(db, model, id)
    if instance:
        for key, value in data.items():
            setattr(instance, key, value)
        await db.commit()
        await db.refresh(instance)
    return instance


async def delete(db: AsyncSession, model: Type[T], id: Any) -> bool:
    """Delete a record by ID."""
    instance = await get_one(db, model, id)
    if instance:
        await db.delete(instance)
        await db.commit()
        return True
    return False
