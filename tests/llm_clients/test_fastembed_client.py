import pytest
from unittest.mock import patch
import numpy as np

from kavalai.llm_clients.fastembed_client import FastEmbedClient


@pytest.fixture
def fastembed_client():
    return FastEmbedClient()


@pytest.mark.asyncio
async def test_fastembed_compute_embeddings_unit():
    """Unit test for compute_embeddings with mocking."""
    client = FastEmbedClient()
    texts = ["This is a test document.", "Another test document."]
    model = "BAAI/bge-small-en-v1.5"

    mock_embeddings = [np.array([0.1] * 384), np.array([0.2] * 384)]

    with patch(
        "kavalai.llm_clients.fastembed_client.TextEmbedding"
    ) as MockTextEmbedding:
        mock_instance = MockTextEmbedding.return_value
        mock_instance.embed.return_value = iter(mock_embeddings)

        embeddings, stats = await client.compute_embeddings(
            model=model, texts=texts, normalize=False
        )

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1] * 384
        assert embeddings[1] == [0.2] * 384
        assert stats.call_type == "embedding"
        assert stats.model == "fastembed/bge-small-en-v1.5"
        assert stats.batch_size == 2
        assert stats.cost == 0.0

        MockTextEmbedding.assert_called_once_with(
            model_name=model, cache_dir=None, threads=None
        )
        mock_instance.embed.assert_called_once_with(texts)


@pytest.mark.asyncio
async def test_fastembed_list_models_unit():
    """Unit test for list_models with mocking."""
    client = FastEmbedClient()

    mock_supported_models = [
        {"model": "BAAI/bge-small-en-v1.5"},
        {"model": "BAAI/bge-base-en-v1.5"},
    ]

    with patch(
        "kavalai.llm_clients.fastembed_client.TextEmbedding"
    ) as MockTextEmbedding:
        MockTextEmbedding.list_supported_models.return_value = mock_supported_models

        models = await client.list_models()
        assert models == ["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5"]


@pytest.mark.asyncio
async def test_fastembed_normalization():
    """Unit test for normalization in compute_embeddings."""
    client = FastEmbedClient()
    texts = ["test"]
    model = "BAAI/bge-small-en-v1.5"

    mock_embeddings = [np.array([1.0, 1.0])]

    with patch(
        "kavalai.llm_clients.fastembed_client.TextEmbedding"
    ) as MockTextEmbedding:
        mock_instance = MockTextEmbedding.return_value
        mock_instance.embed.return_value = iter(mock_embeddings)

        # Test with normalize=True
        embeddings, _ = await client.compute_embeddings(
            model=model, texts=texts, normalize=True
        )

        # L2 normalization of [1.0, 1.0] is [1/sqrt(2), 1/sqrt(2)]
        expected = [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]
        assert np.allclose(embeddings[0], expected)
