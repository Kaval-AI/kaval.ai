from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from kavalai.db import (
    User,
    Project,
    ProjectMembership,
    ProjectRole,
    insert,
    update,
    delete,
    get_one,
    get_all,
    is_member,
    is_owner,
    get_user_projects,
)


@pytest.mark.asyncio
async def test_crud_utilities(db: AsyncSession):
    """Test the generic CRUD utility functions in db.py."""
    # Test Insert
    user_data = {"email": "utility@test.com", "name": "Utility Test"}
    user = await insert(db, User, user_data)
    assert user.id is not None
    assert user.email == "utility@test.com"

    # Test Get One
    fetched_user = await get_one(db, User, user.id)
    assert fetched_user.name == "Utility Test"

    # Test Update
    updated_user = await update(db, User, user.id, {"name": "New Name"})
    assert updated_user.name == "New Name"

    # Test Get All
    all_users = await get_all(db, User)
    assert len(all_users) >= 1

    # Test Delete
    success = await delete(db, User, user.id)
    assert success is True
    deleted_user = await get_one(db, User, user.id)
    assert deleted_user is None


@pytest.mark.asyncio
async def test_access_control_logic(db: AsyncSession):
    """Test is_member and is_owner helper functions."""
    # Setup
    user = await insert(db, User, {"email": "acl@test.com", "name": "ACL User"})
    project = await insert(db, Project, {"name": "ACL Project"})

    # Add membership as owner
    await insert(
        db,
        ProjectMembership,
        {"user_id": user.id, "project_id": project.id, "role": ProjectRole.owner},
    )

    # Test helpers
    assert await is_member(db, user.id, project.id) is True
    assert await is_owner(db, user.id, project.id) is True

    # Test non-existent member
    assert await is_member(db, uuid4(), project.id) is False


@pytest.mark.asyncio
async def test_get_user_projects_with_role(db: AsyncSession):
    # Setup
    user = await insert(db, User, {"email": "role@test.com", "name": "Role User"})
    project = await insert(db, Project, {"name": "Role Project"})
    await insert(
        db,
        ProjectMembership,
        {"user_id": user.id, "project_id": project.id, "role": ProjectRole.viewer},
    )

    # Execute
    results = await get_user_projects(db, user.id)

    # Assert
    assert len(results) == 1
    assert results[0]["name"] == "Role Project"
    assert results[0]["role"] == "viewer"


@pytest.mark.asyncio
async def test_cascade_delete(db: AsyncSession):
    """Ensure deleting a project removes its memberships but not the users."""
    user = await insert(db, User, {"email": "cascade@test.com", "name": "Cascade"})
    project = await insert(db, Project, {"name": "Delete Me"})

    await insert(
        db,
        ProjectMembership,
        {"user_id": user.id, "project_id": project.id, "role": ProjectRole.owner},
    )

    # Delete project
    await delete(db, Project, project.id)

    # Check that membership is gone (via cascade)
    membership = await get_one(db, ProjectMembership, (user.id, project.id))
    assert membership is None

    # Check that user still exists
    still_here = await get_one(db, User, user.id)
    assert still_here is not None
