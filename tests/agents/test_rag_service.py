import pytest

from kavalai.agents.rag_service import RagService, RagServiceResult


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
        texts, source_metadata, collection_name="test_coll"
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
        texts, metadata, collection_name=collection, source_ids=source_ids
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
        texts, metadata, collection_name=collection, source_ids=source_ids
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
        texts, metadata, collection_name=collection, source_ids=source_ids
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
