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
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kavalai.agents.db import RagIndex
from kavalai.backoffice.db import Project, ProjectCache
from kavalai.backoffice.embedding_projector import train_pca
from kavalai.llm_clients.streamer import Streamer


@pytest.mark.asyncio
async def test_train_pca(
    backoffice_db: AsyncSession,
    agents_db: AsyncSession,
    agents_session_maker,
    postgres_container,
):
    # Setup: Create project in backoffice
    project_name = "test_project"
    project = Project(name=project_name)
    backoffice_db.add(project)
    await backoffice_db.commit()
    await backoffice_db.refresh(project)

    # Setup: Add embeddings to agents DB
    collection = "test_collection"
    items = [
        RagIndex(
            model="test_model",
            collection_name=collection,
            source_id=f"label{i}",
            content=f"content{i}",
            embedding_size=3,
            embedding=[float(i), 1.0, 0.5],
        )
        for i in range(10)
    ]
    agents_db.add_all(items)
    await agents_db.commit()

    # Setup: Streamer
    streamer = Streamer(stream_delta=True)
    value_streamer = streamer.get_value_streamer("pca_streamer")

    # Need session makers for the function
    # backoffice_db is a session, but train_pca expects a session_maker
    # We can create a mock session maker that returns the existing session
    # However, train_pca uses 'async with session_maker() as session:'
    # So we need something that behaves like that.

    class MockSessionMaker:
        def __init__(self, session):
            self.session = session

        def __call__(self):
            return self

        async def __aenter__(self):
            return self.session

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    bo_sm = MockSessionMaker(backoffice_db)
    agents_sm = agents_session_maker  # This is already a sessionmaker

    # Execute
    await train_pca(
        bo_session_maker=bo_sm,
        agents_session_maker=agents_sm,
        project_name=project_name,
        collection_name=collection,
        streamer=value_streamer,
    )

    # Verify: Check project_cache
    stmt = select(ProjectCache).where(ProjectCache.project_id == project.id)
    result = await backoffice_db.execute(stmt)
    cache_entries = result.scalars().all()

    assert len(cache_entries) == 2
    names = [c.name for c in cache_entries]
    assert f"pca_model_{collection}" in names
    assert f"pca_sample_train_data_{collection}" in names

    # Verify model
    model_entry = next(c for c in cache_entries if c.name == f"pca_model_{collection}")
    model_bytes = base64.b64decode(model_entry.value)
    pca_model = pickle.loads(model_bytes)  # nosec B301
    assert pca_model.n_components == 2

    # Verify sample data
    sample_entry = next(
        c for c in cache_entries if c.name == f"pca_sample_train_data_{collection}"
    )
    sample_data = json.loads(sample_entry.value)
    assert len(sample_data) == 10
    assert "label" in sample_data[0]
    assert "x" in sample_data[0]
    assert "y" in sample_data[0]

    # Verify streamer messages (all chunks are queued; iterate until 'complete')
    messages = []
    types = []
    async for chunk in streamer:
        messages.append(chunk.value)
        types.append(chunk.type)

    assert any("Starting PCA training" in m for m in messages if m)
    assert any("Downloading embeddings" in m for m in messages if m)
    assert any("Computing PCA model" in m for m in messages if m)
    assert any("Generating sample points" in m for m in messages if m)
    assert any("Storing results in cache" in m for m in messages if m)
    assert any("Finished downloading 10/10 items" in m for m in messages if m)
    assert types[-1] == "complete"
    assert messages[-2] == "PCA training completed successfully."


@pytest.mark.asyncio
async def test_train_pca_project_not_found(
    backoffice_db: AsyncSession, agents_session_maker
):
    class MockSessionMaker:
        def __init__(self, session):
            self.session = session

        def __call__(self):
            return self

        async def __aenter__(self):
            return self.session

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with pytest.raises(ValueError, match="Project 'non_existent' not found"):
        await train_pca(
            bo_session_maker=MockSessionMaker(backoffice_db),
            agents_session_maker=agents_session_maker,
            project_name="non_existent",
            collection_name="test",
        )


@pytest.mark.asyncio
async def test_train_pca_endpoint(
    backoffice_db: AsyncSession,
    agents_db: AsyncSession,
    agents_session_maker,
    postgres_container,
    monkeypatch,
):
    from kavalai.backoffice import server
    from kavalai.backoffice.server import app
    from httpx import AsyncClient, ASGITransport

    # Setup project and data
    project = Project(name="test_project_endpoint")
    backoffice_db.add(project)
    await backoffice_db.commit()
    await backoffice_db.refresh(project)

    collection = "test_collection_endpoint"
    items = [
        RagIndex(
            model="test_model",
            collection_name=collection,
            source_id=f"label{i}",
            content=f"content{i}",
            embedding_size=3,
            embedding=[float(i), 1.0, 0.5],
        )
        for i in range(10)
    ]
    agents_db.add_all(items)
    await agents_db.commit()

    # Mock authentication and project access
    monkeypatch.setattr(server, "assert_logged_in", lambda r: None)

    async def mock_get_project_and_assert_access(request, project_id):
        return project

    monkeypatch.setattr(
        server, "get_project_and_assert_access", mock_get_project_and_assert_access
    )

    # Mock get_backoffice_session and get_project_session
    class MockSessionMaker:
        def __init__(self, session):
            self.session = session

        def __call__(self):
            return self

        async def __aenter__(self):
            return self.session

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    monkeypatch.setattr(
        server, "get_backoffice_session", lambda: MockSessionMaker(backoffice_db)
    )
    monkeypatch.setattr(
        server, "get_project_session", lambda p: MockSessionMaker(agents_db)
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Use a timeout to avoid hanging if SSE fails
        response = await ac.get(
            f"/projects/{project.id}/rag/train-pca?collection_name={collection}",
            timeout=10,
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    # Verify that messages were streamed
    lines = response.text.splitlines()
    data_lines = [line[5:] for line in lines if line.startswith("data:")]
    messages = [json.loads(line) for line in data_lines]

    values = [m["value"] for m in messages if m.get("value")]
    assert any("Starting PCA training" in v for v in values)
    assert any("Finished downloading 10/10 items" in v for v in values)
    assert any("Processed final" in v for v in values)
    assert any("completed successfully" in v for v in values)
    assert messages[-1]["type"] == "complete"

    # Verify project cache
    stmt = select(ProjectCache).where(ProjectCache.project_id == project.id)
    result = await backoffice_db.execute(stmt)
    cache_entries = result.scalars().all()
    assert len(cache_entries) == 2
