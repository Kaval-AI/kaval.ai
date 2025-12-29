from uuid import UUID

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from kavalai.db import User, Project, ProjectMembership, ProjectRole


@pytest.mark.asyncio
async def test_create_user(db: AsyncSession):
    """Test that a user can be persisted and retrieved."""
    new_user = User(
        email="dev@kavalai.io",
        name="Kavalai Developer",
        is_admin=True
    )
    db.add(new_user)
    await db.commit()

    # Retrieve and verify
    stmt = select(User).where(User.email == "dev@kavalai.io")
    result = await db.execute(stmt)
    user = result.scalars().first()

    assert user is not None
    assert user.name == "Kavalai Developer"
    assert isinstance(user.id, UUID)
    assert user.is_admin is True


@pytest.mark.asyncio
async def test_create_project(db: AsyncSession):
    """Test project creation and default timestamps."""
    project = Project(
        name="Internal Tooling",
        description="A project for internal automation"
    )
    db.add(project)
    await db.commit()

    stmt = select(Project).where(Project.name == "Internal Tooling")
    result = await db.execute(stmt)
    fetched = result.scalars().first()

    assert fetched.description == "A project for internal automation"
    assert fetched.created_at is not None


@pytest.mark.asyncio
async def test_project_membership_workflow(db: AsyncSession):
    """
    Test the full relationship:
    User -> ProjectMembership -> Project
    """
    # 1. Setup User and Project
    owner = User(email="owner@test.com", name="Owner User")
    viewer = User(email="viewer@test.com", name="Viewer User")
    project = Project(name="Shared Project")

    db.add_all([owner, viewer, project])
    await db.flush()

    m1 = ProjectMembership(user_id=owner.id, project_id=project.id, role=ProjectRole.owner)
    m2 = ProjectMembership(user_id=viewer.id, project_id=project.id, role=ProjectRole.viewer)

    db.add_all([m1, m2])
    await db.commit()

    # 3. Verify Relationship from Project Side
    stmt = select(Project).options(selectinload(Project.members)).where(Project.id == project.id)
    res = await db.execute(stmt)
    fetched_project = res.scalars().first()

    assert len(fetched_project.members) == 2

    # 4. Verify Relationship from User Side
    stmt = select(User).options(selectinload(User.memberships)).where(User.id == owner.id)
    res = await db.execute(stmt)
    fetched_owner = res.scalars().first()

    assert len(fetched_owner.memberships) == 1
    assert fetched_owner.memberships[0].role == ProjectRole.owner
