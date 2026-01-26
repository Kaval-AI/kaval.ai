import uuid
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from kavalai.agents.db import EmbeddingProfile, RagIndex
from kavalai.backoffice import db as bo_db
from kavalai.backoffice.server import app
from kavalai.crud import insert


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_projects_get_embedding_configs(client, backoffice_db, agents_db):
    project_id = uuid.uuid4()

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

    # 2. Setup Embedding Profile in Agents DB
    profile = await insert(
        agents_db,
        EmbeddingProfile,
        {
            "name": "Test Embedding",
            "provider": "openai",
            "model_name": "text-embedding-3-small",
            "api_key": "secret-key",
            "embedding_size": 1536,
            "config": {"key": "secret"},
        },
    )

    # 3. Mock authentication and database connection
    with patch("kavalai.backoffice.server.assert_logged_in"), patch(
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
        response = await client.get(f"/projects/{project_id}/embedding-configs")

    # 5. Verify results
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    # Find the one we just created
    item = next((d for d in data if d["id"] == str(profile.id)), None)
    assert item is not None
    assert item["name"] == "Test Embedding"
    assert item["model_name"] == "text-embedding-3-small"

    # Ensure sensitive data is NOT present in any item
    for d in data:
        assert "api_key" not in d
        assert "secret-key" not in str(d)
        assert "secret" not in str(d)


@pytest.mark.asyncio
async def test_projects_rag_query(client, backoffice_db, agents_db):
    project_id = uuid.uuid4()

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

    # 2. Setup Embedding Profile in Agents DB
    profile = await insert(
        agents_db,
        EmbeddingProfile,
        {
            "name": "Test Embedding",
            "provider": "openai",
            "model_name": "text-embedding-3-small",
            "api_key": "secret-key",
            "embedding_size": 3,
            "config": {},
        },
    )

    # 3. Add some RAG data
    await insert(
        agents_db,
        RagIndex,
        {
            "embedding_profile_id": profile.id,
            "collection_name": "test-collection",
            "content": "Relevant content",
            "embedding_size": 3,
            "embedding": [0.1, 0.2, 0.3],
            "rag_metadata": {"source": "test"},
        },
    )

    # 4. Mock authentication, compute_embeddings and database connection
    with patch("kavalai.backoffice.server.assert_logged_in"), patch(
        "kavalai.backoffice.server.get_project_and_assert_access", return_value=project
    ), patch(
        "kavalai.agents.db.db_manager.get_sessionmaker"
    ) as mock_get_sessionmaker, patch(
        "kavalai.agents.rag_service.compute_embeddings", new_callable=AsyncMock
    ) as mock_compute:
        mock_compute.return_value = [[0.1, 0.2, 0.3]]

        # Mock the session maker to return our agents_db
        mock_session_context = MagicMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=agents_db)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)
        mock_get_sessionmaker.return_value = MagicMock(
            return_value=mock_session_context
        )

        # 5. Call the endpoint
        query_data = {
            "embedding_profile_id": str(profile.id),
            "text": "Find relevant content",
            "collection_name": "test-collection",
            "top_k": 5,
        }
        response = await client.post(
            f"/projects/{project_id}/rag/query", json=query_data
        )

    # 6. Verify results
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["content"] == "Relevant content"
    assert data[0]["collection_name"] == "test-collection"
    assert data[0]["rag_metadata"] == {"source": "test"}
