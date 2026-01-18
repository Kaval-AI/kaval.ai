import os
import math
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from kavalai.llm_clients.openai import OpenAIClient
from kavalai.llm_clients.gemini import GeminiClient
from kavalai.llm_clients.common import compute_embeddings
from kavalai.agents.db import LLMProfile


@pytest.mark.asyncio
async def test_openai_compute_embeddings_mock():
    client = OpenAIClient(api_key="fake-key")

    mock_response = MagicMock()
    mock_data = [
        MagicMock(embedding=[0.1, 0.2, 0.3]),
        MagicMock(embedding=[0.4, 0.5, 0.6]),
    ]
    mock_response.data = mock_data

    client.client.embeddings.create = AsyncMock(return_value=mock_response)

    texts = ["hello", "world"]
    model = "text-embedding-3-small"

    # Test without normalization
    embeddings = await client.compute_embeddings(model=model, texts=texts)

    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]
    client.client.embeddings.create.assert_called_once_with(input=texts, model=model)

    # Test with normalization
    client.client.embeddings.create.reset_mock()
    embeddings_norm = await client.compute_embeddings(
        model=model, texts=texts, normalize=True
    )

    assert len(embeddings_norm) == 2
    # Check if normalized
    for emb in embeddings_norm:
        norm = math.sqrt(sum(x * x for x in emb))
        assert math.isclose(norm, 1.0, rel_tol=1e-9)


@pytest.mark.asyncio
async def test_gemini_compute_embeddings_mock():
    with patch("google.genai.Client"):
        client = GeminiClient(api_key="fake-key")

        mock_response = MagicMock()
        mock_response.embeddings = [
            MagicMock(values=[0.1, 0.2, 0.3]),
            MagicMock(values=[0.4, 0.5, 0.6]),
        ]

        client.client.aio.models.embed_content = AsyncMock(return_value=mock_response)

        texts = ["hello", "world"]
        model = "text-embedding-004"

        # Test without normalization
        embeddings = await client.compute_embeddings(model=model, texts=texts)

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]

        # Test with normalization
        embeddings_norm = await client.compute_embeddings(
            model=model, texts=texts, normalize=True
        )
        assert len(embeddings_norm) == 2
        for emb in embeddings_norm:
            norm = math.sqrt(sum(x * x for x in emb))
            assert math.isclose(norm, 1.0, rel_tol=1e-9)


@pytest.mark.asyncio
async def test_common_compute_embeddings_mock():
    llm_profile = LLMProfile(
        id=1,
        name="test-profile",
        provider="openai",
        model_name="text-embedding-3-small",
        api_key="fake-key",
    )

    with patch("kavalai.llm_clients.common.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.compute_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_get_client.return_value = mock_client

        texts = ["hello"]
        result = await compute_embeddings(llm_profile, texts, normalize=True)

        assert result == [[0.1, 0.2, 0.3]]
        mock_client.compute_embeddings.assert_called_once_with(
            model="text-embedding-3-small", texts=texts, normalize=True
        )


# Integration tests (skipped if no API key)
@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_openai_embeddings_integration():
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAIClient(api_key=api_key)

    texts = ["This is a test document.", "Another test document."]
    model = "text-embedding-3-small"

    embeddings = await client.compute_embeddings(
        model=model, texts=texts, normalize=True
    )

    assert len(embeddings) == 2
    for emb in embeddings:
        assert len(emb) == 1536
        norm = math.sqrt(sum(x * x for x in emb))
        assert math.isclose(norm, 1.0, rel_tol=1e-5)
