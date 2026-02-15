import asyncio
import os

import pytest
from pydantic import BaseModel

from unittest.mock import AsyncMock, patch, MagicMock
from openai import LengthFinishReasonError
from openai.types.responses import ResponseTextDeltaEvent, ResponseCompletedEvent
from kavalai.llm_clients.openai_client import OpenAIClient
from kavalai.llm_clients.common import Streamer


class SimpleResponse(BaseModel):
    answer: str
    confidence: float


@pytest.mark.asyncio
async def test_openai_chat_completions_streaming():
    client = OpenAIClient(api_key="fake-key")

    mock_stream = AsyncMock()

    # Use MagicMock for chunks to avoid pydantic validation errors
    chunk1 = MagicMock(spec=ResponseTextDeltaEvent)
    chunk1.delta = '{"answer": "He'

    chunk2 = MagicMock(spec=ResponseTextDeltaEvent)
    chunk2.delta = 'llo", "confid'

    chunk3 = MagicMock(spec=ResponseTextDeltaEvent)
    chunk3.delta = 'ence": 1.0}'

    chunk4 = MagicMock(spec=ResponseCompletedEvent)
    chunk4.response = MagicMock()
    chunk4.response.usage = MagicMock()
    chunk4.response.usage.input_tokens = 10
    chunk4.response.usage.output_tokens = 20

    # Simulate a stream of chunks
    mock_stream.__aiter__.return_value = [chunk1, chunk2, chunk3, chunk4]

    mock_stream_manager = MagicMock()
    mock_stream_manager.__aenter__.return_value = mock_stream
    mock_stream_manager.__aexit__ = AsyncMock(return_value=None)

    queue = asyncio.Queue()
    streamer = Streamer(name="test", queue=queue)

    with patch.object(
        client.client.responses, "stream", return_value=mock_stream_manager
    ) as _:
        result, stats = await client.chat_completions(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            response_model=SimpleResponse,
            streamer=streamer,
        )

    assert isinstance(result, SimpleResponse)
    assert result.answer == "Hello"
    assert result.confidence == 1.0
    assert stats.prompt_tokens == 10
    assert stats.completion_tokens == 20
    assert stats.total_tokens == 30

    # Check queue content.
    stream_out = ""
    while not queue.empty():
        stream_out += await queue.get() + "\n"

    assert len(stream_out) > 0
    assert '"partial"' in stream_out
    assert '"complete"' in stream_out
    assert "Hello" in stream_out


@pytest.mark.asyncio
async def test_openai_chat_completion_no_retry_in_client():
    client = OpenAIClient(api_key="fake-key")

    mock_stream_manager = MagicMock()
    mock_stream_manager.__aenter__.side_effect = LengthFinishReasonError(
        completion=MagicMock()
    )

    with patch.object(
        client.client.responses, "stream", return_value=mock_stream_manager
    ):
        with pytest.raises(LengthFinishReasonError):
            await client.chat_completions(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                response_model=SimpleResponse,
            )


@pytest.mark.asyncio
async def test_openai_chat_completion_with_service_tier():
    client = OpenAIClient(api_key="fake-key", service_tier="priority")

    mock_stream = AsyncMock()

    chunk1 = MagicMock(spec=ResponseTextDeltaEvent)
    chunk1.delta = '{"answer": "ok", "confidence": 1.0}'

    chunk2 = MagicMock(spec=ResponseCompletedEvent)
    chunk2.response = MagicMock()
    chunk2.response.usage = MagicMock()
    chunk2.response.usage.input_tokens = 5
    chunk2.response.usage.output_tokens = 5

    mock_stream.__aiter__.return_value = [chunk1, chunk2]

    mock_stream_manager = MagicMock()
    mock_stream_manager.__aenter__.return_value = mock_stream
    mock_stream_manager.__aexit__ = AsyncMock(return_value=None)

    with patch.object(
        client.client.responses, "stream", return_value=mock_stream_manager
    ) as mock_stream_method:
        await client.chat_completions(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            response_model=SimpleResponse,
        )
        assert mock_stream_method.call_args.kwargs["service_tier"] == "priority"


@pytest.mark.asyncio
async def test_openai_compute_embeddings_with_service_tier():
    client = OpenAIClient(api_key="fake-key", service_tier="priority")
    mock_create = AsyncMock()

    mock_data = MagicMock()
    mock_data.embedding = [0.1]
    mock_response = MagicMock()
    mock_response.data = [mock_data]
    mock_response.usage.total_tokens = 5
    mock_response.model_dump.return_value = {}

    mock_create.return_value = mock_response

    with patch.object(client.client.embeddings, "create", mock_create):
        await client.compute_embeddings(model="text-embedding-3-small", texts=["hi"])
        assert "service_tier" not in mock_create.call_args.kwargs


@pytest.mark.asyncio
async def test_openai_compute_embeddings():
    client = OpenAIClient(api_key="fake-key")
    mock_create = AsyncMock()

    mock_data = MagicMock()
    mock_data.embedding = [0.1, 0.2]
    mock_response = MagicMock()
    mock_response.data = [mock_data]
    mock_response.usage.total_tokens = 5
    mock_response.model_dump.return_value = {"fake": "resp"}

    mock_create.return_value = mock_response

    with patch.object(client.client.embeddings, "create", mock_create):
        embeddings, stats = await client.compute_embeddings(
            model="text-embedding-3-small", texts=["hi"], normalize=True
        )

    assert embeddings[0] == pytest.approx(
        [0.1 / (0.1**2 + 0.2**2) ** 0.5, 0.2 / (0.1**2 + 0.2**2) ** 0.5]
    )
    assert stats.total_tokens == 5
    assert mock_create.call_count == 1


@pytest.mark.asyncio
async def test_openai_compute_embeddings_zero_norm():
    client = OpenAIClient(api_key="fake-key")
    mock_create = AsyncMock()

    mock_data = MagicMock()
    mock_data.embedding = [0.0, 0.0]
    mock_response = MagicMock()
    mock_response.data = [mock_data]
    mock_response.usage.total_tokens = 0
    mock_response.model_dump.return_value = {}

    mock_create.return_value = mock_response

    with patch.object(client.client.embeddings, "create", mock_create):
        embeddings, stats = await client.compute_embeddings(
            model="text-embedding-3-small", texts=["hi"], normalize=True
        )

    assert embeddings[0] == pytest.approx([0.0, 0.0])


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_openai_structured_output_stream():
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAIClient(api_key=api_key)

    messages = [
        {
            "role": "user",
            "content": "What is 2+2? Answer in JSON format with 'answer' and 'confidence' fields.",
        }
    ]
    queue = asyncio.Queue()
    streamer = Streamer(name="test", queue=queue)
    content, stats = await client.chat_completions(
        model="gpt-4o-mini",
        messages=messages,
        response_model=SimpleResponse,
        streamer=streamer,
    )

    assert isinstance(content, SimpleResponse)
    assert "4" in content.answer
    assert content.confidence >= 0.0
    assert not queue.empty()
