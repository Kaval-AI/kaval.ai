import uuid
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from kavalai.backoffice import db as bo_db
from kavalai.agents.db import LLMProfile
from kavalai.backoffice.server import app
from kavalai.crud import insert

# Import fixtures from agents tests


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_projects_get_llm_configs(client, backoffice_db, agents_db):
    project_id = uuid.uuid4()
    user_id = str(uuid.uuid4())

    # 1. Setup Project in Backoffice DB
    project = bo_db.Project(
        id=project_id,
        name="Test Project",
        db_user="user",
        db_password="password",
        db_host="localhost",
        db_port=5432,
        db_name="test_db",
    )
    backoffice_db.add(project)
    await backoffice_db.commit()

    # 2. Setup LLM Profile in Agents DB
    profile = await insert(
        agents_db,
        LLMProfile,
        {
            "name": "Test LLM",
            "provider": "openai",
            "model_name": "gpt-4",
            "api_key": "secret-key",
            "credentials": {"key": "secret"},
        },
    )

    # 3. Mock authentication and database connection
    with patch("kavalai.backoffice.server.is_logged_in", return_value=True), patch(
        "kavalai.backoffice.server.assert_logged_in"
    ), patch(
        "starlette.requests.Request.session", {"user_info": {"id": user_id}}
    ), patch(
        "kavalai.backoffice.server.get_project_and_assert_access", return_value=project
    ), patch("kavalai.agents.db.db_manager.get_sessionmaker") as mock_get_sessionmaker:
        # Mock the session maker to return our agents_db
        mock_session_context = MagicMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=agents_db)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)
        mock_get_sessionmaker.return_value = MagicMock(
            return_value=mock_session_context
        )

    # 4. Call the endpoint
    response = await client.get(f"/projects/{project_id}/llm-configs")

    # 5. Verify results
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    # Find the one we just created
    item = next((d for d in data if d["id"] == str(profile.id)), None)
    assert item is not None
    assert item["name"] == "Test LLM"
    assert item["provider"] == "openai"
    assert item["model_name"] == "gpt-4"

    # Ensure sensitive data is NOT present in any item
    for d in data:
        assert "api_key" not in d
        assert "credentials" not in d
        assert "secret-key" not in str(d)
        assert "secret" not in str(d)
