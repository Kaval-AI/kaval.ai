"""
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import base64
import json
import pickle
import uuid
from unittest.mock import patch, MagicMock, AsyncMock

import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport
from sklearn.decomposition import IncrementalPCA

from kavalai.backoffice import db
from kavalai.backoffice.server import app


@pytest.mark.asyncio
async def test_projects_rag_query_with_pca(backoffice_db, agents_db, monkeypatch):
    from kavalai.backoffice import server

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
    backoffice_db.add(project)

    # Setup PCA model in cache
    collection = "test_col"
    ipca = IncrementalPCA(n_components=2)
    # Fit with some dummy data
    ipca.partial_fit(np.random.rand(10, 3))

    model_data = base64.b64encode(pickle.dumps(ipca)).decode("utf-8")
    cache_model = db.ProjectCache(
        project_id=project_id, name=f"pca_model_{collection}", value=model_data
    )

    samples = [
        {"label": "sample1", "x": 0.1, "y": 0.2},
        {"label": "sample2", "x": 0.3, "y": 0.4},
    ]
    cache_samples = db.ProjectCache(
        project_id=project_id,
        name=f"pca_sample_train_data_{collection}",
        value=json.dumps(samples),
    )

    backoffice_db.add_all([cache_model, cache_samples])
    await backoffice_db.commit()

    # Mock authentication
    monkeypatch.setattr(server, "assert_logged_in", lambda r: None)
    monkeypatch.setattr(
        server, "get_project_and_assert_access", AsyncMock(return_value=project)
    )

    # Mock session factories
    class MockSessionMaker:
        def __init__(self, session):
            self.session = session

        def __call__(self):
            return self

        async def __aenter__(self):
            return self.session

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(
        server, "get_backoffice_session", lambda: MockSessionMaker(backoffice_db)
    )
    monkeypatch.setattr(
        server, "get_project_session", lambda p: MockSessionMaker(agents_db)
    )

    # Mock RagService
    mock_rag_result = MagicMock()
    mock_rag_result.id = uuid.uuid4()
    mock_rag_result.content = "result content"
    mock_rag_result.similarity = 0.9

    # We need to mock the RagService.query return value and also handle the internal embedding_client call
    with patch(
        "kavalai.backoffice.server.PostgresRagService"
    ) as mock_rag_service_class:
        mock_instance = mock_rag_service_class.return_value
        mock_instance.query = AsyncMock(return_value=[mock_rag_result])
        mock_instance.embedding_client.compute_embeddings = AsyncMock(
            return_value=([np.random.rand(3).tolist()], None)
        )

        # The endpoint fetches result embeddings through the service
        # (storage is backend-owned).
        mock_instance.get_embeddings_by_ids = AsyncMock(
            return_value={mock_rag_result.id: [0.1, 0.2, 0.3]}
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            query_data = {
                "model": "test_model",
                "text": "test query",
                "collection_name": collection,
                "top_k": 5,
            }
            response = await ac.post(
                f"/projects/{project_id}/rag/query", json=query_data
            )

            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert "pca_data" in data
            assert data["pca_data"] is not None
            assert "query" in data["pca_data"]
            assert "results" in data["pca_data"]
            assert "samples" in data["pca_data"]
            assert len(data["pca_data"]["samples"]) == 2
            assert data["pca_data"]["results"][0]["label"] == "result content"
