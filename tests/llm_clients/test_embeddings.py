import os
import pytest
from kavalai.llm_clients.llm_client import LLMClient


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_llm_client_compute_embeddings_openai():
    """Test cross-provider embedding generation via LLMClient for OpenAI."""
    client = LLMClient(model="openai/text-embedding-3-small")
    texts = ["This is a test document."]
    embeddings, stats = await client.compute_embeddings(texts=texts, normalize=True)

    assert len(embeddings) == 1
    assert len(embeddings[0]) == 1536
    assert stats.call_type == "embedding"
    assert stats.model == "openai/text-embedding-3-small"


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
async def test_llm_client_compute_embeddings_gemini():
    """Test cross-provider embedding generation via LLMClient for Gemini."""
    # We use a fixed model name here, LLMClient will handle it
    client = LLMClient(model="gemini/gemini-embedding-001")
    texts = ["This is a test document."]

    # This should now succeed.
    embeddings, stats = await client.compute_embeddings(texts=texts)
    assert len(embeddings) == 1
    assert len(embeddings[0]) > 0
    assert stats.call_type == "embedding"
    assert stats.model == "gemini/gemini-embedding-001"
