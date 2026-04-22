import asyncio
import os
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from pydantic import BaseModel

from kavalai.llm_clients.common import Streamer, StreamContent
from kavalai.llm_clients.openai_client import OpenAIClient


class SimpleResponse(BaseModel):
    answer: str
    confidence: float


@pytest.fixture
def openai_client():
    return OpenAIClient()


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_openai_chat_completions_simple(openai_client):
    """Test simple message and string result."""
    messages = [{"role": "user", "content": "Say 'Hello'"}]
    result, stats = await openai_client.chat_completions(
        model="gpt-4o-mini", messages=messages
    )

    assert "Hello" in result
    assert stats.prompt_tokens > 0
    assert stats.completion_tokens > 0
    assert stats.model == "openai/gpt-4o-mini"
    assert stats.cost is None
    assert stats.currency is None


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_openai_chat_completions_formatted(openai_client):
    """Test formatted result (structured output)."""
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    result, stats = await openai_client.chat_completions(
        model="gpt-4o-mini", messages=messages, response_model=SimpleResponse
    )

    assert isinstance(result, SimpleResponse)
    assert "Paris" in result.answer
    assert result.confidence > 0
    assert stats.prompt_tokens > 0
    assert stats.completion_tokens > 0


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_openai_chat_completions_multimodal(openai_client):
    """Test results with text and image input."""
    # Small 1x1 black PNG
    dummy_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{dummy_image_b64}"},
                },
            ],
        }
    ]
    result, stats = await openai_client.chat_completions(
        model="gpt-4o-mini", messages=messages
    )

    assert isinstance(result, str)
    assert len(result) > 0
    assert stats.prompt_tokens > 0
    assert stats.completion_tokens > 0


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_openai_chat_completions_streamer(openai_client):
    """Test streamer functionality."""
    messages = [{"role": "user", "content": "Say 'Hi'"}]
    queue = asyncio.Queue()
    streamer = Streamer("test_streamer", queue)

    result, stats = await openai_client.chat_completions(
        model="gpt-4o-mini", messages=messages, streamer=streamer
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

    assert "Hi" in result
    assert len(partials) > 0
    assert final_streamed == result


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_openai_stream_delta(openai_client):
    """Test stream_delta functionality."""
    messages = [{"role": "user", "content": "Say 'Hello'"}]

    # Test delta=False (default)
    queue = asyncio.Queue()
    streamer = Streamer("test_no_delta", queue)
    await openai_client.chat_completions(
        model="gpt-4o-mini", messages=messages, streamer=streamer, stream_delta=False
    )
    partials_no_delta = []
    while not queue.empty():
        c = StreamContent.model_validate_json(await queue.get())
        if c.type == "partial":
            partials_no_delta.append(c.value)

    if len(partials_no_delta) > 1:
        assert partials_no_delta[1].startswith(partials_no_delta[0])

    # Test delta=True
    queue = asyncio.Queue()
    streamer = Streamer("test_delta", queue)
    await openai_client.chat_completions(
        model="gpt-4o-mini", messages=messages, streamer=streamer, stream_delta=True
    )
    partials_delta = []
    while not queue.empty():
        c = StreamContent.model_validate_json(await queue.get())
        if c.type == "partial":
            partials_delta.append(c.value)

    if len(partials_delta) > 1 and partials_delta[0]:
        assert not partials_delta[1].startswith(partials_delta[0])


@pytest.mark.asyncio
async def test_openai_compute_embeddings_unit(openai_client):
    """Unit test for compute_embeddings with mocking."""
    texts = ["This is a test document.", "Another test document."]
    model = "text-embedding-3-small"

    # Mock response from OpenAI API
    mock_data = [MagicMock(embedding=[0.1] * 1536), MagicMock(embedding=[0.2] * 1536)]
    mock_usage = MagicMock(total_tokens=10)
    mock_response = MagicMock(data=mock_data, usage=mock_usage)
    mock_response_dict = {
        "data": [{"embedding": [0.1] * 1536}, {"embedding": [0.2] * 1536}],
        "usage": {"total_tokens": 10},
    }
    mock_response.model_dump.return_value = mock_response_dict

    with patch.object(
        openai_client.client.embeddings, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_response

        embeddings, stats = await openai_client.compute_embeddings(
            model=model, texts=texts, normalize=False
        )

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1] * 1536
        assert embeddings[1] == [0.2] * 1536
        assert stats.call_type == "embedding"
        assert stats.model == f"openai/{model}"
        assert stats.total_tokens == 10
        assert stats.cost is None
        assert stats.response_data == mock_response_dict

        mock_create.assert_called_once_with(
            input=texts, model=model, timeout=openai_client.timeout
        )


@pytest.mark.asyncio
async def test_openai_compute_embeddings_normalize_unit(openai_client):
    """Unit test for compute_embeddings with normalization."""
    texts = ["test"]
    model = "text-embedding-3-small"

    mock_data = [MagicMock(embedding=[1.0] * 1536)]
    mock_response = MagicMock(
        data=mock_data,
        usage=MagicMock(total_tokens=5),
        model_dump=MagicMock(
            return_value={
                "data": [{"embedding": [1.0] * 1536}],
                "usage": {"total_tokens": 5},
            }
        ),
    )

    with patch.object(
        openai_client.client.embeddings, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_response
        # We pass an explicit normalizer to avoid global state from get_default_normalizer()
        from kavalai.normalizer import Normalizer

        explicit_normalizer = Normalizer(l2=True)
        embeddings, _ = await openai_client.compute_embeddings(
            model=model, texts=texts, normalize=True, normalizer=explicit_normalizer
        )

        assert len(embeddings[0]) == 1536  # Assert embedding dimension


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_openai_compute_embeddings(openai_client):
    """Test embedding generation."""
    texts = ["This is a test document.", "Another test document."]
    model = "text-embedding-3-small"

    embeddings, _ = await openai_client.compute_embeddings(
        model=model, texts=texts, normalize=True
    )

    assert len(embeddings) == 2  # Assert number of embeddings
    assert len(embeddings[1]) == 1536  # Assert embedding dimension


@pytest.mark.asyncio
async def test_openai_service_tier_unit(openai_client):
    """Unit test for service_tier handling."""
    from openai.types.responses import (
        ResponseTextDeltaEvent,
    )

    messages = [{"role": "user", "content": "test"}]
    model = "gpt-4o"

    # Mock response
    mock_event = MagicMock()
    mock_event.delta = "test"
    mock_event.type = "text_delta"
    # To satisfy isinstance(event, ResponseTextDeltaEvent)

    # We use a hack to make isinstance work with MagicMock if needed,
    # but actually we can just use the real class if we provide all fields,
    # OR we can mock the isinstance check.
    # Actually, let's just use a MagicMock and patch isinstance in the test context if possible,
    # or just use the real class with minimal fields if it allows it.
    # Since it's Pydantic, let's just use MagicMock and patch where it's used.

    with patch(
        "kavalai.llm_clients.openai_client.isinstance",
        side_effect=lambda obj, cls: True
        if cls == ResponseTextDeltaEvent
        else isinstance(obj, cls),
    ):
        # Mock stream context manager
        mock_stream = MagicMock()
        mock_stream.__aenter__.return_value = AsyncMock()
        mock_stream.__aenter__.return_value.__aiter__.return_value = [mock_event]

        # We need to mock the response usage at the end if we want to avoid the UnboundLocalError
        # But for this test, we mostly care about the call to the responses.stream method
        with patch(
            "openai.resources.responses.AsyncResponses.stream", return_value=mock_stream
        ) as mock_stream_call:
            # 1. Test from constructor
            client_with_tier = OpenAIClient(service_tier="priority")
            await client_with_tier.chat_completions(model=model, messages=messages)

            _, kwargs = mock_stream_call.call_args
            assert kwargs.get("service_tier") == "priority"

            # 2. Test from kwargs (overriding constructor)
            mock_stream_call.reset_mock()
            await client_with_tier.chat_completions(
                model=model, messages=messages, service_tier="flex"
            )

            _, kwargs = mock_stream_call.call_args
            assert kwargs.get("service_tier") == "flex"

            # 3. Test priority mapping via LLMClient
            from kavalai.llm_clients.llm_client import LLMClient

            llm_client = LLMClient("openai/gpt-4o")
            llm_client.client = client_with_tier
            mock_stream_call.reset_mock()
            await llm_client.chat_completions(messages=messages, priority="high")
            _, kwargs = mock_stream_call.call_args
            assert kwargs.get("service_tier") == "priority"
