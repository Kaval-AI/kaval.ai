import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from kavalai.agents.rag_service import RagService, RagServiceResult
from kavalai.normalizer import Normalizer
from kavalai.agents.db import ModelCallStat


@pytest.fixture
def embedding_model():
    return "openai/text-embedding-3-small"


@pytest.mark.asyncio
async def test_rag_service_indexing(
    agents_db_config, migrated_agents_db, embedding_model
):
    service = RagService(agents_db_config["uri"], embedding_model)

    texts = ["hello", "world"]
    source_metadata = [{"id": 1}, {"id": 2}]

    items = await service.batch_index(
        texts=texts, metadata_list=source_metadata, collection_name="test_coll"
    )
    assert len(items) == 2
    assert items[0].collection_name == "test_coll"
    assert items[0].content == "hello"
    assert len(items[0].embedding) == 1536
    assert items[0].rag_metadata == {"id": 1}
    assert items[1].content == "world"
    assert len(items[1].embedding) == 1536

    result = await service.query("hello", top_k=1, collection_name="test_coll")
    assert len(result) == 1
    assert isinstance(result[0], RagServiceResult)
    assert result[0].content == "hello"
    assert result[0].similarity > 0


@pytest.mark.asyncio
async def test_rag_service_deletion(
    agents_db_config, migrated_agents_db, embedding_model
):
    service = RagService(agents_db_config["uri"], embedding_model)

    texts = ["item 1", "item 2", "item 3"]
    source_ids = ["sid1", "sid2", "sid3"]
    metadata = [{}, {}, {}]
    collection = "delete_test"

    # Index items
    await service.batch_index(
        texts=texts,
        metadata_list=metadata,
        collection_name=collection,
        source_ids=source_ids,
    )

    # Verify they exist
    results = await service.query("item", top_k=10, collection_name=collection)
    assert len(results) == 3

    # Delete sid1 and sid3
    await service.delete_by_source_ids(collection, ["sid1", "sid3"])

    # Verify only sid2 remains
    results = await service.query("item", top_k=10, collection_name=collection)
    assert len(results) == 1
    assert results[0].source_id == "sid2"
    assert results[0].content == "item 2"


@pytest.mark.asyncio
async def test_rag_service_keep_best(
    agents_db_config, migrated_agents_db, embedding_model
):
    service = RagService(agents_db_config["uri"], embedding_model)

    # We index multiple items with the same source_id
    texts = ["apple", "apple pie", "banana"]
    source_ids = ["fruit_1", "fruit_1", "fruit_2"]
    metadata = [{}, {}, {}]
    collection = "keep_best_test"

    await service.batch_index(
        texts=texts,
        metadata_list=metadata,
        collection_name=collection,
        source_ids=source_ids,
    )

    # Query for "apple".
    # Without keep_best (default False), we should get both fruit_1 items
    results = await service.query("apple", top_k=10, collection_name=collection)
    assert len(results) == 3

    # Now query with keep_best=True
    results_best = await service.query(
        "apple", top_k=10, collection_name=collection, keep_best=True
    )

    # Should only have one result per source_id
    assert len(results_best) == 2  # fruit_1 and fruit_2
    source_ids_found = [r.source_id for r in results_best]
    assert len(source_ids_found) == len(set(source_ids_found))

    # "apple" should be better than "apple pie" for the query "apple"
    fruit_1_best = [r for r in results_best if r.source_id == "fruit_1"]
    assert len(fruit_1_best) == 1
    assert fruit_1_best[0].content == "apple"


@pytest.mark.asyncio
async def test_rag_service_with_normalizer():
    # Mock database session
    mock_session = AsyncMock()
    mock_session.execute.return_value = MagicMock(all=MagicMock(return_value=[]))

    # Mock session maker
    mock_session_maker = MagicMock()
    mock_session_maker.return_value.__aenter__.return_value = mock_session

    model = "openai/text-embedding-3-small"
    normalizer = Normalizer(l2=True)

    # We need to mock compute_embeddings in rag_service
    with patch("kavalai.agents.rag_service.compute_embeddings") as mock_compute:
        mock_stats = MagicMock(spec=ModelCallStat)
        mock_compute.return_value = ([[0.1, 0.2, 0.3]], mock_stats)

        # Initialize RagService with normalizer
        # Pass a session instead of URI to avoid real DB engine creation
        service = RagService(
            uri_or_session=mock_session, model=model, normalizer=normalizer
        )

        assert service.normalizer == normalizer

        # 1. Test batch_index
        await service.batch_index(
            texts=["test"], metadata_list=[{}], collection_name="test_coll"
        )

        mock_compute.assert_called_with(
            model=model, texts=["test"], normalizer=normalizer
        )

        # 2. Test query
        mock_compute.reset_mock()
        await service.query("test query")

        mock_compute.assert_called_with(
            model=model, texts=["test query"], normalizer=normalizer
        )

        # 3. Test compute_similarity_matrix
        mock_compute.reset_mock()
        await service.compute_similarity_matrix(texts=["t1"], source_ids=["s1"])

        mock_compute.assert_called_with(
            model=model, texts=["t1"], normalizer=normalizer
        )


@pytest.mark.asyncio
async def test_rag_service_without_normalizer():
    # Mock database session
    mock_session = AsyncMock()
    mock_session.execute.return_value = MagicMock(all=MagicMock(return_value=[]))

    model = "openai/text-embedding-3-small"

    with patch("kavalai.agents.rag_service.compute_embeddings") as mock_compute:
        mock_stats = MagicMock(spec=ModelCallStat)
        mock_compute.return_value = ([[0.1, 0.2, 0.3]], mock_stats)

        # Initialize RagService without normalizer
        service = RagService(uri_or_session=mock_session, model=model)

        assert service.normalizer is None

        await service.query("test query")

        mock_compute.assert_called_with(
            model=model, texts=["test query"], normalizer=None
        )


@pytest.mark.asyncio
async def test_rag_service_keep_best_duplicates(
    agents_db_config, migrated_agents_db, embedding_model
):
    """Test that keep_best handles duplicate content/distances correctly."""
    service = RagService(agents_db_config["uri"], embedding_model)

    collection = "duplicate_test"
    # Index exactly the same content for the same source_id multiple times
    texts = ["Tesla Model 3 is an electric car", "Tesla Model 3 is an electric car"]
    source_ids = ["tesla_3", "tesla_3"]
    metadata = [{"brand": "Tesla"}, {"brand": "Tesla"}]

    await service.batch_index(
        texts=texts,
        metadata_list=metadata,
        collection_name=collection,
        source_ids=source_ids,
    )

    # Query with keep_best=True
    results = await service.query(
        "Tesla electric car", top_k=10, collection_name=collection, keep_best=True
    )

    # Should only have one result despite multiple identical best matches
    assert len(results) == 1
    assert results[0].source_id == "tesla_3"


@pytest.mark.asyncio
async def test_rag_service_top_k_with_keep_best(
    agents_db_config, migrated_agents_db, embedding_model
):
    service = RagService(agents_db_config["uri"], embedding_model)

    # Index items for 5 different source IDs, each with 2 items
    texts = []
    source_ids = []
    for i in range(1, 6):
        sid = f"source_{i}"
        texts.extend([f"content {i} a", f"content {i} b"])
        source_ids.extend([sid, sid])

    metadata = [{} for _ in texts]
    collection = "top_k_keep_best_test"

    await service.batch_index(
        texts=texts,
        metadata_list=metadata,
        collection_name=collection,
        source_ids=source_ids,
    )

    # Query with top_k=3 and keep_best=True
    # We expect exactly 3 results, each from a different source_id
    top_k = 3
    results = await service.query(
        "content", top_k=top_k, collection_name=collection, keep_best=True
    )

    assert len(results) == top_k
    unique_source_ids = set(r.source_id for r in results)
    assert len(unique_source_ids) == top_k

    # Now query with top_k=1 and keep_best=True
    # This might fail if the implementation is buggy (e.g. limit applied before join)
    results_1 = await service.query(
        "content", top_k=1, collection_name=collection, keep_best=True
    )
    assert len(results_1) == 1


@pytest.mark.asyncio
async def test_rag_service_query_source_ids(
    agents_db_config, migrated_agents_db, embedding_model
):
    service = RagService(agents_db_config["uri"], embedding_model)

    texts = ["apple", "banana", "cherry"]
    source_ids = ["sid_apple", "sid_banana", "sid_cherry"]
    metadata = [{}, {}, {}]
    collection = "source_ids_test"

    await service.batch_index(
        texts=texts,
        metadata_list=metadata,
        collection_name=collection,
        source_ids=source_ids,
    )

    # Query with filtering by source_ids
    results = await service.query(
        "fruit",
        top_k=10,
        collection_name=collection,
        source_ids=["sid_apple", "sid_cherry"],
    )

    assert len(results) == 2
    found_ids = {r.source_id for r in results}
    assert found_ids == {"sid_apple", "sid_cherry"}
    assert "sid_banana" not in found_ids


@pytest.mark.asyncio
async def test_rag_service_query_source_ids_with_keep_best(
    agents_db_config, migrated_agents_db, embedding_model
):
    service = RagService(agents_db_config["uri"], embedding_model)

    # Index multiple items for some source_ids
    texts = ["apple 1", "apple 2", "banana 1", "banana 2", "cherry"]
    source_ids = ["sid_apple", "sid_apple", "sid_banana", "sid_banana", "sid_cherry"]
    metadata = [{}, {}, {}, {}, {}]
    collection = "source_ids_keep_best_test"

    await service.batch_index(
        texts=texts,
        metadata_list=metadata,
        collection_name=collection,
        source_ids=source_ids,
    )

    # Query with filtering by source_ids and keep_best=True
    results = await service.query(
        "fruit",
        top_k=10,
        collection_name=collection,
        source_ids=["sid_apple", "sid_banana"],
        keep_best=True,
    )

    # We expect exactly 2 results: one for sid_apple and one for sid_banana
    assert len(results) == 2
    found_ids = {r.source_id for r in results}
    assert found_ids == {"sid_apple", "sid_banana"}
    assert "sid_cherry" not in found_ids


@pytest.mark.asyncio
async def test_compute_similarity_matrix(
    agents_db_config, migrated_agents_db, embedding_model
):
    service = RagService(agents_db_config["uri"], embedding_model)

    # Index some documents
    texts = ["apple", "apple pie", "banana", "banana bread", "cherry"]
    source_ids = ["fruit_1", "fruit_1", "fruit_2", "fruit_2", "fruit_3"]
    await service.batch_index(
        texts=texts,
        metadata_list=[{}] * len(texts),
        collection_name="matrix_test",
        source_ids=source_ids,
    )

    # Compute similarity matrix
    queries = ["apple", "banana"]
    target_source_ids = ["fruit_1", "fruit_2", "fruit_3", "fruit_nonexistent"]

    # Test "min" method (shortest distance = max similarity)
    matrix_min = await service.compute_similarity_matrix(
        texts=queries, source_ids=target_source_ids, method="min"
    )

    assert len(matrix_min) == 2  # 2 queries
    assert len(matrix_min[0]) == 4  # 4 target source ids
    assert len(matrix_min[1]) == 4

    # matrix_min[0][0] is similarity between "apple" and "fruit_1" (contains "apple", "apple pie")
    # "apple" vs "apple" should have high similarity
    assert matrix_min[0][0] > 0.9
    # matrix_min[0][1] is similarity between "apple" and "fruit_2" (contains "banana", "banana bread")
    # should be lower
    assert matrix_min[0][0] > matrix_min[0][1]
    # nonexistent source should have 0 similarity
    assert matrix_min[0][3] == 0.0

    # Test "avg" method
    matrix_avg = await service.compute_similarity_matrix(
        texts=queries, source_ids=target_source_ids, method="avg"
    )
    assert len(matrix_avg) == 2
    assert len(matrix_avg[0]) == 4

    # For "fruit_1", "apple" query vs ["apple", "apple pie"]
    # min distance will be "apple" vs "apple" (distance near 0)
    # avg distance will be average of ("apple" vs "apple") and ("apple" vs "apple pie")
    # So avg similarity should be less than or equal to min similarity
    assert matrix_avg[0][0] <= matrix_min[0][0]
