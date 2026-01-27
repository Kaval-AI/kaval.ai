import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from kavalai.agents.db import EmbeddingProfile, RagIndex
from kavalai.crud import insert, get_one


@pytest.mark.asyncio
async def test_embedding_profile_and_rag_index(agents_db: AsyncSession):
    """Test EmbeddingProfile and RagIndex models and their relationship."""
    # 1. Create Embedding Profile
    profile = await insert(
        agents_db,
        EmbeddingProfile,
        {
            "name": "OpenAI Embeddings",
            "provider": "openai",
            "model_name": "text-embedding-3-small",
            "api_key": "sk-test",
        },
    )
    assert profile.name == "OpenAI Embeddings"

    # 2. Create RAG Index item linked to profile
    rag_item = await insert(
        agents_db,
        RagIndex,
        {
            "embedding_profile_id": profile.id,
            "collection_name": "default",
            "source_id": "test-1",
            "embedding": [0.1] * 1536,
            "embedding_size": 1536,
            "content": "This is a test document for RAG.",
            "rag_metadata": {"source": "test_suite"},
        },
    )
    assert rag_item.embedding_profile_id == profile.id
    assert rag_item.source_id == "test-1"
    assert len(rag_item.embedding) == 1536
    assert rag_item.content == "This is a test document for RAG."

    # 3. Create RAG Index item with JSON content
    json_item = await insert(
        agents_db,
        RagIndex,
        {
            "embedding_profile_id": profile.id,
            "collection_name": "default",
            "source_id": "test-2",
            "embedding": [0.5] * 768,
            "embedding_size": 768,
            "content": '{"key": "value", "nested": [1, 2, 3]}',
        },
    )
    assert json_item.content == '{"key": "value", "nested": [1, 2, 3]}'

    # 4. Create RAG Index item with binary content
    binary_item = await insert(
        agents_db,
        RagIndex,
        {
            "embedding_profile_id": profile.id,
            "collection_name": "default",
            "source_id": "test-3",
            "embedding": [0.9] * 384,
            "embedding_size": 384,
            "content": "binary data",
        },
    )
    assert binary_item.content == "binary data"

    # 5. Test Cascade Delete (EmbeddingProfile -> RagIndex)
    profile_id = profile.id
    rag_item_id = rag_item.id
    json_item_id = json_item.id
    binary_item_id = binary_item.id

    # Action: Delete the profile
    await agents_db.delete(profile)
    await agents_db.commit()
    agents_db.expire_all()

    # Asserts - Profile and all linked RAG items should be gone
    assert await get_one(agents_db, EmbeddingProfile, profile_id) is None
    assert await get_one(agents_db, RagIndex, rag_item_id) is None
    assert await get_one(agents_db, RagIndex, json_item_id) is None
    assert await get_one(agents_db, RagIndex, binary_item_id) is None
