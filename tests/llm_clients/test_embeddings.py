from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kavalai.agents.db import ModelCallStat
from kavalai.llm_clients import embeddings as emb
from kavalai.llm_clients.embeddings import (
    BaseEmbeddingClient,
    FastEmbedClient,
    GeminiEmbeddingClient,
    OllamaEmbeddingClient,
    OpenAIEmbeddingClient,
    make_embedding_client,
)
from kavalai.normalizer import Normalizer


# ------------------------------------------------------------------- factory
def test_make_embedding_client_providers(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert isinstance(
        make_embedding_client("openai/text-embedding-3-small"), OpenAIEmbeddingClient
    )
    assert isinstance(
        make_embedding_client("gemini/text-embedding-004"), GeminiEmbeddingClient
    )
    assert isinstance(
        make_embedding_client("ollama/nomic-embed-text"), OllamaEmbeddingClient
    )
    fc = make_embedding_client("fastembed/BAAI/bge-small-en-v1.5")
    assert isinstance(fc, FastEmbedClient)
    # The provider is split off but the rest of the path is kept intact.
    assert fc.model == "BAAI/bge-small-en-v1.5"


def test_make_embedding_client_errors():
    with pytest.raises(ValueError, match="provider/model"):
        make_embedding_client("no-slash")
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        make_embedding_client("anthropic/x")


def test_base_client_not_implemented():
    import asyncio

    with pytest.raises(NotImplementedError):
        asyncio.run(BaseEmbeddingClient("m").compute_embeddings(["a"]))


# ---------------------------------------------------------------- providers
async def test_openai_embeddings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    client = OpenAIEmbeddingClient("text-embedding-3-small")
    response = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2])],
        usage=SimpleNamespace(total_tokens=7),
        model_dump=lambda: {"ok": True},
    )
    client.client = MagicMock()
    client.client.embeddings.create = AsyncMock(return_value=response)

    vectors, stats = await client.compute_embeddings(["hello"])
    assert vectors == [[0.1, 0.2]]
    assert isinstance(stats, ModelCallStat)
    assert stats.call_type == "embedding"
    assert stats.model == "openai/text-embedding-3-small"
    assert stats.total_tokens == 7


async def test_openai_embeddings_normalized(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    client = OpenAIEmbeddingClient("text-embedding-3-small")
    response = SimpleNamespace(
        data=[SimpleNamespace(embedding=[3.0, 4.0])],
        usage=SimpleNamespace(total_tokens=1),
        model_dump=lambda: {},
    )
    client.client = MagicMock()
    client.client.embeddings.create = AsyncMock(return_value=response)

    vectors, _ = await client.compute_embeddings(
        ["x"], normalize=True, normalizer=Normalizer(l2=True)
    )
    # L2-normalised [3,4] -> [0.6, 0.8]
    assert vectors[0][0] == pytest.approx(0.6)
    assert vectors[0][1] == pytest.approx(0.8)


async def test_gemini_embeddings(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    client = GeminiEmbeddingClient("text-embedding-004")
    response = SimpleNamespace(embeddings=[SimpleNamespace(values=[0.5, 0.6])])
    client.client = MagicMock()
    client.client.aio.models.embed_content = AsyncMock(return_value=response)
    # Avoid importing the real google types in compute_embeddings.
    monkeypatch.setattr(
        "google.genai.types.EmbedContentConfig", lambda **kw: kw, raising=False
    )

    vectors, stats = await client.compute_embeddings(["hi"])
    assert vectors == [[0.5, 0.6]]
    assert stats.model == "gemini/text-embedding-004"


async def test_ollama_embeddings():
    client = OllamaEmbeddingClient("nomic-embed-text")
    client.client = MagicMock()
    client.client.embed = AsyncMock(
        return_value={"embeddings": [[0.1, 0.2]], "prompt_eval_count": 4}
    )
    vectors, stats = await client.compute_embeddings(["a", "b"])
    assert vectors == [[0.1, 0.2], [0.1, 0.2]]  # one per input text
    assert stats.total_tokens == 8
    assert stats.model == "ollama/nomic-embed-text"


async def test_fastembed_embeddings(monkeypatch):
    client = FastEmbedClient("BAAI/bge-small-en-v1.5")

    class _Vec:
        def tolist(self):
            return [0.9, 0.1]

    fake_model = MagicMock()
    fake_model.embed = lambda texts, **kw: [_Vec() for _ in texts]
    monkeypatch.setattr(client, "_get_model", lambda: fake_model)

    vectors, stats = await client.compute_embeddings(["a"])
    assert vectors == [[0.9, 0.1]]
    assert stats.call_type == "embedding"
    assert stats.model == "fastembed/bge-small-en-v1.5"
    assert stats.currency == "USD"


def test_fastembed_get_model_lazy(monkeypatch):
    # _get_model builds a TextEmbedding lazily and caches it.
    created = []

    class _Fake:
        def __init__(self, **kw):
            created.append(kw)

    monkeypatch.setattr(emb, "FastEmbedClient", FastEmbedClient)
    monkeypatch.setitem(
        __import__("sys").modules,
        "fastembed",
        SimpleNamespace(TextEmbedding=_Fake),
    )
    client = FastEmbedClient("m", cache_dir="/tmp", threads=2)
    first = client._get_model()
    second = client._get_model()
    assert first is second
    assert created and created[0]["model_name"] == "m"
