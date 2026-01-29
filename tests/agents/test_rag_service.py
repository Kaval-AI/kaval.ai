import pytest

from kavalai.agents.rag_service import RagService, RagServiceResult


@pytest.fixture
def embedding_model():
    return "text-embedding-3-small"


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
