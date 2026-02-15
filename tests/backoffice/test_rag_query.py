import uuid
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from httpx import AsyncClient, ASGITransport

from kavalai.backoffice import db
from kavalai.backoffice.server import app


@pytest.fixture
def client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_projects_rag_query_with_source_ids(client):
    project_id = uuid.uuid4()
    project = db.Project(
        id=project_id,
        name="Test Project",
        db_user="user",
        db_password="password",
        db_host="localhost",
        db_port=5432,
        db_name="test_db",
    )

    query_data = {
        "model": "text-embedding-3-small",
        "text": "test query",
        "collection_name": "test_collection",
        "top_k": 5,
        "source_ids": ["source1", "source2"],
    }

    with patch("kavalai.backoffice.server.assert_logged_in"), patch(
        "kavalai.backoffice.server.get_project_and_assert_access", return_value=project
    ), patch("kavalai.agents.db.db_manager.get_sessionmaker") as mock_sm, patch(
        "kavalai.backoffice.server.RagService"
    ) as mock_rag_service_class:
        mock_session = AsyncMock()
        mock_sm.return_value = MagicMock(return_value=mock_session)
        mock_session.__aenter__.return_value = mock_session

        mock_rag_service_instance = AsyncMock()
        mock_rag_service_class.return_value = mock_rag_service_instance
        mock_rag_service_instance.query.return_value = [{"content": "result"}]

        response = await client.post(
            f"/projects/{project_id}/rag/query", json=query_data
        )

        assert response.status_code == 200
        assert response.json() == [{"content": "result"}]

        mock_rag_service_instance.query.assert_called_once_with(
            text="test query",
            top_k=5,
            collection_name="test_collection",
            source_ids=["source1", "source2"],
        )
