import pytest
from unittest.mock import patch, AsyncMock
from kavalai.agents.db import EmbeddingProfile, RagIndex
from kavalai.agents.rag_service import RagService
from kavalai.crud import insert


@pytest.mark.asyncio
async def test_rag_service_indexing(agents_db):
    # 1. Setup profile
    profile = await insert(
        agents_db,
        EmbeddingProfile,
        {
            "name": "Test Embedder",
            "provider": "openai",
            "model_name": "text-embedding-3-small",
        },
    )

    service = RagService(agents_db, profile)

    # 2. Mock compute_embeddings_with_stats
    mock_embeddings = [[0.1] * 1536, [0.2] * 1536]

    with patch(
        "kavalai.agents.rag_service.compute_embeddings_with_stats",
        new_callable=AsyncMock,
    ) as mock_compute:
        mock_compute.return_value = mock_embeddings

        # 3. Test batch_index
        texts = ["hello", "world"]
        source_metadata = [{"id": 1}, {"id": 2}]

        items = await service.batch_index(
            texts, source_metadata, collection_name="test_coll"
        )
        assert len(items) == 2
        assert items[0].collection_name == "test_coll"
        assert items[0].content == "hello"
        assert items[0].embedding == [0.1] * 1536
        assert items[0].rag_metadata == {"id": 1}
        assert items[1].content == "world"
        assert items[1].embedding == [0.2] * 1536

        mock_compute.assert_called_once_with(
            llm_profile=profile, texts=texts, session=agents_db, agent_id=None
        )


@pytest.mark.asyncio
async def test_rag_service_query(agents_db):
    # 1. Setup profile
    profile = await insert(
        agents_db,
        EmbeddingProfile,
        {
            "name": "Test Embedder",
            "provider": "openai",
            "model_name": "text-embedding-3-small",
        },
    )

    # 2. Add some data
    await insert(
        agents_db,
        RagIndex,
        {
            "embedding_profile_id": profile.id,
            "collection_name": "index1",
            "source_id": "match",
            "embedding": [1.0] + [0.0] * 1535,
            "embedding_size": 1536,
            "content": "match",
            "rag_metadata": {"foo": "bar"},
        },
    )
    await insert(
        agents_db,
        RagIndex,
        {
            "embedding_profile_id": profile.id,
            "collection_name": "index2",
            "source_id": "wrong",
            "embedding": [1.0] + [0.0] * 1535,
            "embedding_size": 1536,
            "content": "wrong index",
        },
    )
    await insert(
        agents_db,
        RagIndex,
        {
            "embedding_profile_id": profile.id,
            "collection_name": "index1",
            "source_id": "other",
            "embedding": [1.0] + [0.0] * 1535,
            "embedding_size": 1536,
            "content": "other",
        },
    )

    service = RagService(agents_db, profile)

    # 3. Mock query embedding
    with patch(
        "kavalai.agents.rag_service.compute_embeddings_with_stats",
        new_callable=AsyncMock,
    ) as mock_compute:
        mock_compute.return_value = [[1.0] + [0.0] * 1535]

        # No filters
        results = await service.query("some query")
        assert len(results) == 3

        # Filter by index
        results = await service.query("some query", collection_name="index1")
        assert len(results) == 2
        for r in results:
            assert r["collection_name"] == "index1"

        # Match specific content
        results = await service.query("some query", collection_name="index1")
        assert any(r["content"] == "match" for r in results)
        assert any(r["content"] == "other" for r in results)
        assert results[0]["collection_name"] == "index1"


@pytest.mark.asyncio
async def test_rag_service_indexing_with_dim(agents_db):
    profile = await insert(
        agents_db,
        EmbeddingProfile,
        {
            "name": "Test Embedder",
            "provider": "openai",
            "model_name": "text-embedding-3-small",
        },
    )
    service = RagService(agents_db, profile)

    with patch(
        "kavalai.agents.rag_service.compute_embeddings_with_stats",
        new_callable=AsyncMock,
    ) as mock_compute:
        mock_compute.return_value = [[0.1] * 100]

        await service.batch_index(["test"], [{}])
        # Should work fine now as it's just a VECTOR column
