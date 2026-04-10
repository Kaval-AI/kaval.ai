import os
import pytest
import httpx
from kavalai.llm_clients.ollama_client import OllamaClient
from kavalai.llm_clients.llm_client import LLMClient

OLLAMA_HOST = os.getenv("OLLAMA_HOST")


def is_ollama_running():
    if not OLLAMA_HOST:
        return False

    # Simple check to see if Ollama is up
    url = (
        f"http://{OLLAMA_HOST}/api/tags"
        if "://" not in OLLAMA_HOST
        else f"{OLLAMA_HOST}/api/tags"
    )
    try:
        response = httpx.get(url, timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not is_ollama_running(),
    reason="OLLAMA_HOST not set or Ollama service not reachable",
)


@pytest.fixture
def model_name():
    return "llama3.2:1b"


@pytest.fixture
def embedding_model_name():
    return "nomic-embed-text-v2-moe"


@pytest.mark.asyncio
async def test_ollama_list_models_integration():
    client = OllamaClient()
    models = await client.list_models()
    assert isinstance(models, list)
    # We don't assert specific models as it depends on local setup


@pytest.mark.asyncio
async def test_ollama_chat_completions_integration(model_name):
    client = LLMClient(f"ollama/{model_name}")

    # First check if model is pulled
    models = await client.list_models()
    if not any(model_name in m for m in models):
        pytest.skip(f"Model {model_name} not pulled in Ollama")

    messages = [{"role": "user", "content": "I say 'ping', you say ...?"}]
    content, stats = await client.chat_completions(messages=messages)

    assert isinstance(content, str)
    assert "pong" in content.lower()
    assert stats.model == f"ollama/{model_name}"
    assert stats.prompt_tokens > 0


@pytest.mark.asyncio
async def test_ollama_compute_embeddings_integration(embedding_model_name):
    # Use LLMClient for embeddings too
    client = LLMClient(f"ollama/{embedding_model_name}")

    # Check if model is pulled
    models = await client.list_models()
    if not any(embedding_model_name in m for m in models):
        pytest.skip(f"Model {embedding_model_name} not pulled in Ollama")

    texts = ["Hello world", "Kaval AI is awesome"]
    embeddings, stats = await client.compute_embeddings(texts=texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0
    assert stats.total_tokens > 0
    assert stats.batch_size == 2
