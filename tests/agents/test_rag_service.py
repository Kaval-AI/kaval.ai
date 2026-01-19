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

    # 2. Mock compute_embeddings
    mock_embeddings = [[0.1] * 1536, [0.2] * 1536]

    with patch(
        "kavalai.agents.rag_service.compute_embeddings", new_callable=AsyncMock
    ) as mock_compute:
        mock_compute.return_value = mock_embeddings

        # 3. Test batch_index
        texts = ["hello", "world"]
        metadata = [{"id": 1}, {"id": 2}]

        items = await service.batch_index(texts, metadata)
        assert len(items) == 2
        assert items[0].embedding_profile_name == "Test Embedder"
        assert items[0].text_content == "hello"
        assert items[0].embedding_1536 == [0.1] * 1536
        assert items[0].metadata_ == {"id": 1}
        assert items[1].text_content == "world"
        assert items[1].embedding_1536 == [0.2] * 1536

        mock_compute.assert_called_once_with(llm_profile=profile, texts=texts)


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
            "embedding_profile_name": "index1",
            "embedding_1536": [1.0] + [0.0] * 1535,
            "text_content": "match",
            "mime_type": "text/plain",
        },
    )
    await insert(
        agents_db,
        RagIndex,
        {
            "embedding_profile_id": profile.id,
            "embedding_profile_name": "index2",
            "embedding_1536": [1.0] + [0.0] * 1535,
            "text_content": "wrong index",
            "mime_type": "text/plain",
        },
    )
    await insert(
        agents_db,
        RagIndex,
        {
            "embedding_profile_id": profile.id,
            "embedding_profile_name": "index1",
            "embedding_1536": [1.0] + [0.0] * 1535,
            "text_content": "other",
            "mime_type": "text/plain",
        },
    )

    service = RagService(agents_db, profile)

    # 3. Mock query embedding
    with patch(
        "kavalai.agents.rag_service.compute_embeddings", new_callable=AsyncMock
    ) as mock_compute:
        mock_compute.return_value = [[1.0] + [0.0] * 1535]

        # No filters
        results = await service.query("some query")
        assert len(results) == 3

        # Filter by index
        results = await service.query("some query", index="index1")
        assert len(results) == 2
        for r in results:
            assert r.embedding_profile_name == "index1"

        # Match specific content
        results = await service.query("some query", index="index1")
        assert any(r.text_content == "match" for r in results)
        assert any(r.text_content == "other" for r in results)
        assert results[0].embedding_profile_name == "index1"


@pytest.mark.asyncio
async def test_rag_service_unsupported_dim(agents_db):
    profile = await insert(
        agents_db,
        EmbeddingProfile,
        {
            "name": "Test Embedder",
            "provider": "openai",
            "model_name": "unsupported",
        },
    )
    service = RagService(agents_db, profile)

    with patch(
        "kavalai.agents.rag_service.compute_embeddings", new_callable=AsyncMock
    ) as mock_compute:
        mock_compute.return_value = [[0.1] * 100]  # Dim 100 not supported

        with pytest.raises(ValueError, match="Unsupported embedding dimension: 100"):
            await service.batch_index(["test"], [{}])
