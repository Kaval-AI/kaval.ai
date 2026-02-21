import math
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kavalai.agents.db import ModelCallStat
from kavalai.llm_clients.llm_client import LLMClient
from kavalai.llm_clients.gemini_client import GeminiClient
from kavalai.llm_clients.openai_client import OpenAIClient


@pytest.mark.asyncio
async def test_openai_compute_embeddings_mock():
    client = OpenAIClient(api_key="fake-key")

    mock_response = MagicMock()
    mock_response.usage.total_tokens = 0
    mock_data = [
        MagicMock(embedding=[0.1, 0.2, 0.3]),
        MagicMock(embedding=[0.4, 0.5, 0.6]),
    ]
    mock_response.data = mock_data
    mock_response.model_dump.return_value = {"mock": "response"}

    client.client.embeddings.create = AsyncMock(return_value=mock_response)

    texts = ["hello", "world"]
    model = "text-embedding-3-small"

    # Test without normalization
    embeddings, stats = await client.compute_embeddings(model=model, texts=texts)

    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]
    assert stats.total_tokens == 0
    assert stats.model == f"openai/{model}"
    client.client.embeddings.create.assert_called_once_with(
        input=texts, model=model, timeout=30.0
    )

    # Test with normalization
    client.client.embeddings.create.reset_mock()
    embeddings_norm, stats_norm = await client.compute_embeddings(
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
        mock_response.usage_metadata.total_token_count = 10

        client.client.aio.models.embed_content = AsyncMock(return_value=mock_response)

        texts = ["hello", "world"]
        model = "text-embedding-004"

        # Test without normalization
        embeddings, stats = await client.compute_embeddings(model=model, texts=texts)

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
        assert stats.total_tokens == 10
        assert stats.model == f"gemini/{model}"

        # Test with normalization
        embeddings_norm, stats_norm = await client.compute_embeddings(
            model=model, texts=texts, normalize=True
        )
        assert len(embeddings_norm) == 2
        for emb in embeddings_norm:
            norm = math.sqrt(sum(x * x for x in emb))
            assert math.isclose(norm, 1.0, rel_tol=1e-9)


@pytest.mark.asyncio
async def test_common_compute_embeddings_mock():
    with patch("kavalai.llm_clients.llm_client.LLMClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_stats = ModelCallStat(
            call_type="embedding",
            model="openai/text-embedding-3-small",
            total_tokens=10,
            cost=0.001,
            response_data={},
        )
        mock_client.compute_embeddings.return_value = ([[0.1, 0.2, 0.3]], mock_stats)
        mock_client_class.return_value = mock_client

        texts = ["hello"]
        client = LLMClient(model="openai/text-embedding-3-small")
        result, stats = await client.compute_embeddings(texts=texts, normalize=True)

        assert result == [[0.1, 0.2, 0.3]]
        assert stats.total_tokens == 10
        mock_client.compute_embeddings.assert_called_once_with(
            texts=texts,
            normalize=True,
            normalizer=None,
        )


# Integration tests (skipped if no API key)
@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_openai_embeddings_integration():
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAIClient(api_key=api_key)

    texts = ["This is a test document.", "Another test document."]
    model = "text-embedding-3-small"

    embeddings, stats = await client.compute_embeddings(
        model=model, texts=texts, normalize=True
    )

    assert len(embeddings) == 2
    for emb in embeddings:
        assert len(emb) == 1536
        norm = math.sqrt(sum(x * x for x in emb))
        assert math.isclose(norm, 1.0, rel_tol=1e-5)


async def test_compute_embeddings_with_stats(agents_db):
    with patch("kavalai.llm_clients.llm_client.LLMClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_stats = ModelCallStat(
            call_type="embedding",
            model="openai/text-embedding-3-small",
            total_tokens=100,
            cost=0.000002,
            response_data={"id": "test-res"},
            response_code=200,
            batch_size=1,
        )
        mock_client.compute_embeddings.return_value = ([[0.1] * 1536], mock_stats)
        mock_client_class.return_value = mock_client

        texts = ["hello"]
        client = LLMClient(model="openai/text-embedding-3-small")
        embeddings, stats = await client.compute_embeddings(texts=texts)

        assert embeddings == [[0.1] * 1536]
        assert stats.total_tokens == 100
        assert float(stats.cost) == 0.000002
        assert stats.batch_size == 1
        assert stats.response_code == 200

        mock_client.compute_embeddings.assert_called_once_with(
            texts=texts,
            normalize=False,
            normalizer=None,
        )
