import asyncio
import json
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from kavalai.llm_clients.common import Streamer, StreamContent
from kavalai.llm_clients.ollama_client import OllamaClient


class SimpleResponse(BaseModel):
    answer: str
    confidence: float


@pytest.fixture
def ollama_client():
    return OllamaClient()


@pytest.mark.asyncio
async def test_ollama_chat_completions_simple():
    """Unit test for simple message and string result with mocking."""
    client = OllamaClient()
    messages = [{"role": "user", "content": "Say 'Hello'"}]

    mock_chunks = [
        {"message": {"role": "assistant", "content": "He"}, "done": False},
        {
            "message": {"role": "assistant", "content": "llo"},
            "done": True,
            "prompt_eval_count": 5,
            "eval_count": 2,
        },
    ]

    async def mock_chat(*args, **kwargs):
        for chunk in mock_chunks:
            yield chunk

    with patch.object(client.client, "chat", side_effect=mock_chat):
        result, stats = await client.chat_completions(
            model="llama3.2:1b", messages=messages
        )

        assert result == "Hello"
        assert stats.prompt_tokens == 5
        assert stats.completion_tokens == 2
        assert stats.model == "ollama/llama3.2:1b"


@pytest.mark.asyncio
async def test_ollama_chat_completions_formatted():
    """Unit test for formatted result (structured output) with mocking."""
    client = OllamaClient()
    messages = [{"role": "user", "content": "What is the capital of France?"}]

    response_dict = {"answer": "Paris", "confidence": 0.99}
    mock_chunks = [
        {
            "message": {"role": "assistant", "content": json.dumps(response_dict)},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 15,
        }
    ]

    async def mock_chat(*args, **kwargs):
        for chunk in mock_chunks:
            yield chunk

    with patch.object(client.client, "chat", side_effect=mock_chat):
        result, stats = await client.chat_completions(
            model="llama3.2:1b", messages=messages, response_model=SimpleResponse
        )

        assert isinstance(result, SimpleResponse)
        assert result.answer == "Paris"
        assert result.confidence == 0.99
        assert stats.prompt_tokens == 10
        assert stats.completion_tokens == 15


@pytest.mark.asyncio
async def test_ollama_chat_completions_streamer():
    """Unit test for streamer functionality with mocking."""
    client = OllamaClient()
    messages = [{"role": "user", "content": "Say 'Hi'"}]
    queue = asyncio.Queue()
    streamer = Streamer("test_streamer", queue)

    mock_chunks = [
        {"message": {"role": "assistant", "content": "H"}, "done": False},
        {"message": {"role": "assistant", "content": "i"}, "done": True},
    ]

    async def mock_chat(*args, **kwargs):
        for chunk in mock_chunks:
            yield chunk

    with patch.object(client.client, "chat", side_effect=mock_chat):
        result, stats = await client.chat_completions(
            model="llama3.2:1b", messages=messages, streamer=streamer
        )

        partials = []
        final_streamed = None
        while not queue.empty():
            content_json = await queue.get()
            content = StreamContent.model_validate_json(content_json)
            if content.type == "partial":
                partials.append(content.value)
            elif content.type == "complete":
                final_streamed = content.value

        assert result == "Hi"
        assert len(partials) > 0
        assert final_streamed == result


@pytest.mark.asyncio
async def test_ollama_compute_embeddings_unit():
    """Unit test for compute_embeddings with mocking."""
    client = OllamaClient()
    texts = ["This is a test document.", "Another test document."]
    model = "nomic-embed-text"

    mock_responses = [
        {"embeddings": [[0.1] * 768], "prompt_eval_count": 5},
        {"embeddings": [[0.2] * 768], "prompt_eval_count": 6},
    ]

    with patch.object(client.client, "embed", side_effect=mock_responses) as mock_embed:
        embeddings, stats = await client.compute_embeddings(
            model=model, texts=texts, normalize=False
        )

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1] * 768
        assert embeddings[1] == [0.2] * 768
        assert stats.call_type == "embedding"
        assert stats.model == f"ollama/{model}"
        assert stats.total_tokens == 11
        assert stats.batch_size == 2

        assert mock_embed.call_count == 2


@pytest.mark.asyncio
async def test_ollama_list_models_unit():
    """Unit test for list_models with mocking."""
    client = OllamaClient()

    mock_response = {
        "models": [{"name": "llama3.2:1b"}, {"name": "nomic-embed-text:latest"}]
    }

    with patch.object(client.client, "list", return_value=mock_response):
        models = await client.list_models()
        assert models == ["llama3.2:1b", "nomic-embed-text:latest"]


@pytest.mark.asyncio
async def test_ollama_list_models_with_objects():
    """Unit test for list_models with object-based responses."""
    client = OllamaClient()

    class MockModel:
        def __init__(self, model):
            self.model = model

    class MockResponse:
        def __init__(self, models):
            self.models = models

    mock_response = MockResponse([MockModel("model1"), MockModel("model2")])

    with patch.object(client.client, "list", return_value=mock_response):
        models = await client.list_models()
        assert models == ["model1", "model2"]
