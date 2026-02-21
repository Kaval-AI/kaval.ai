import asyncio
import os

import pytest
from pydantic import BaseModel

from unittest.mock import AsyncMock, patch, MagicMock
from openai import LengthFinishReasonError
from kavalai.llm_clients.openai_client import OpenAIClient
from kavalai.llm_clients.common import Streamer


class SimpleResponse(BaseModel):
    answer: str
    confidence: float


@pytest.mark.asyncio
async def test_openai_chat_completions_streaming():
    client = OpenAIClient(api_key="fake-key")

    mock_stream = AsyncMock()

    # Use MagicMock for chunks to match the new stream implementation
    chunk1 = MagicMock()
    chunk1.type = "content.delta"
    chunk1.delta = '{"answer": "He'

    chunk2 = MagicMock()
    chunk2.type = "content.delta"
    chunk2.delta = 'llo", "confid'

    chunk3 = MagicMock()
    chunk3.type = "content.delta"
    chunk3.delta = 'ence": 1.0}'

    # Simulate a stream of chunks
    mock_stream.__aiter__.return_value = [chunk1, chunk2, chunk3]

    final_completion = MagicMock()
    final_completion.usage = MagicMock()
    final_completion.usage.prompt_tokens = 10
    final_completion.usage.completion_tokens = 20
    mock_stream.get_final_completion = AsyncMock(return_value=final_completion)

    mock_stream_manager = MagicMock()
    mock_stream_manager.__aenter__.return_value = mock_stream
    mock_stream_manager.__aexit__ = AsyncMock(return_value=None)

    queue = asyncio.Queue()
    streamer = Streamer(name="test", queue=queue)

    with patch.object(
        client.client.beta.chat.completions, "stream", return_value=mock_stream_manager
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

    mock_stream = AsyncMock()
    final_completion = MagicMock()
    final_completion.usage = MagicMock()
    final_completion.usage.prompt_tokens = 5
    final_completion.usage.completion_tokens = 5
    mock_stream.get_final_completion = AsyncMock(return_value=final_completion)

    mock_stream_manager = MagicMock()
    mock_stream_manager.__aenter__.side_effect = LengthFinishReasonError(
        completion=MagicMock()
    )

    with patch.object(
        client.client.beta.chat.completions, "stream", return_value=mock_stream_manager
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

    chunk1 = MagicMock()
    chunk1.type = "content.delta"
    chunk1.delta = '{"answer": "ok", "confidence": 1.0}'

    mock_stream.__aiter__.return_value = [chunk1]

    final_completion = MagicMock()
    final_completion.usage = MagicMock()
    final_completion.usage.prompt_tokens = 5
    final_completion.usage.completion_tokens = 5
    mock_stream.get_final_completion = AsyncMock(return_value=final_completion)

    mock_stream_manager = MagicMock()
    mock_stream_manager.__aenter__.return_value = mock_stream
    mock_stream_manager.__aexit__ = AsyncMock(return_value=None)

    with patch.object(
        client.client.beta.chat.completions, "stream", return_value=mock_stream_manager
    ) as mock_stream_method:
        await client.chat_completions(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            response_model=SimpleResponse,
        )
        assert mock_stream_method.call_args.kwargs["service_tier"] == "priority"


@pytest.mark.asyncio
async def test_openai_chat_completion_with_temperature():
    client = OpenAIClient(api_key="fake-key")

    mock_stream = AsyncMock()

    chunk1 = MagicMock()
    chunk1.type = "content.delta"
    chunk1.delta = '{"answer": "ok", "confidence": 1.0}'

    mock_stream.__aiter__.return_value = [chunk1]

    final_completion = MagicMock()
    final_completion.usage = MagicMock()
    final_completion.usage.prompt_tokens = 5
    final_completion.usage.completion_tokens = 5
    mock_stream.get_final_completion = AsyncMock(return_value=final_completion)

    mock_stream_manager = MagicMock()
    mock_stream_manager.__aenter__.return_value = mock_stream
    mock_stream_manager.__aexit__ = AsyncMock(return_value=None)

    with patch.object(
        client.client.beta.chat.completions, "stream", return_value=mock_stream_manager
    ) as mock_stream_method:
        await client.chat_completions(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            response_model=SimpleResponse,
            temperature=0.7,
        )
        assert mock_stream_method.call_args.kwargs["temperature"] == 0.7


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
    import kavalai.normalizer

    kavalai.normalizer._default_normalizer = None

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
    assert stats.cost == (5 * 0.02) / 1_000_000
    assert mock_create.call_count == 1


@pytest.mark.asyncio
async def test_openai_compute_embeddings_zero_norm():
    import kavalai.normalizer

    kavalai.normalizer._default_normalizer = None

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
async def test_openai_compute_embeddings_custom_normalizer():
    from kavalai.normalizer import Normalizer

    client = OpenAIClient(api_key="fake-key")
    mock_create = AsyncMock()

    mock_data = MagicMock()
    mock_data.embedding = [3.0, 4.0]
    mock_response = MagicMock()
    mock_response.data = [mock_data]
    mock_response.usage.total_tokens = 5
    mock_response.model_dump.return_value = {}

    mock_create.return_value = mock_response

    # L1 normalizer: [3.0, 4.0] -> [3/7, 4/7]
    l1_normalizer = Normalizer(l1=True)

    with patch.object(client.client.embeddings, "create", mock_create):
        embeddings, stats = await client.compute_embeddings(
            model="text-embedding-3-small",
            texts=["hi"],
            normalize=True,
            normalizer=l1_normalizer,
        )

    assert embeddings[0] == pytest.approx([3 / 7, 4 / 7])


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_openai_generate_image():
    client = OpenAIClient(api_key="fake-key")
    mock_generate = AsyncMock()

    mock_data = MagicMock()
    mock_data.b64_json = "fake-base64"
    mock_response = MagicMock()
    mock_response.data = [mock_data]

    mock_generate.return_value = mock_response

    with patch.object(client.client.images, "generate", mock_generate):
        image_base64, stats = await client.generate_image(
            model="dall-e-3", prompt="a cat"
        )

    assert image_base64 == "fake-base64"
    assert stats.call_type == "image_generation"
    assert stats.model == "openai/dall-e-3"


@pytest.mark.asyncio
async def test_openai_list_models():
    client = OpenAIClient(api_key="fake-key")
    mock_list = AsyncMock()

    model1 = MagicMock()
    model1.id = "gpt-4o"
    model2 = MagicMock()
    model2.id = "gpt-4o-mini"

    mock_response = MagicMock()
    mock_response.data = [model1, model2]

    mock_list.return_value = mock_response

    with patch.object(client.client.models, "list", mock_list):
        models = await client.list_models()

    assert "gpt-4o" in models
    assert "gpt-4o-mini" in models
    assert len(models) == 2


@pytest.mark.asyncio
async def test_openai_chat_completions_multimodal():
    client = OpenAIClient(api_key="fake-key")

    mock_stream = AsyncMock()
    # Mock chunks
    chunk1 = MagicMock()
    chunk1.type = "content.delta"
    chunk1.delta = '{"answer": "A cat", "confidence": 0.9}'

    mock_stream.__aiter__.return_value = [chunk1]

    final_completion = MagicMock()
    final_completion.usage = MagicMock()
    final_completion.usage.prompt_tokens = 50
    final_completion.usage.completion_tokens = 10
    mock_stream.get_final_completion = AsyncMock(return_value=final_completion)

    mock_stream_manager = MagicMock()
    mock_stream_manager.__aenter__.return_value = mock_stream
    mock_stream_manager.__aexit__ = AsyncMock(return_value=None)

    messages = [
        {
            "role": "user",
            "content": "What is in this image?",
            "images": ["base64-image-data"],
        }
    ]

    with patch.object(
        client.client.beta.chat.completions, "stream", return_value=mock_stream_manager
    ) as mock_stream_call:
        result, stats = await client.chat_completions(
            model="gpt-4o",
            messages=messages,
            response_model=SimpleResponse,
        )

    # Verify how it was called
    called_messages = mock_stream_call.call_args.kwargs["messages"]
    assert len(called_messages) == 1
    assert called_messages[0]["role"] == "user"
    assert isinstance(called_messages[0]["content"], list)
    assert called_messages[0]["content"][0]["type"] == "text"
    assert called_messages[0]["content"][1]["type"] == "image_url"
    assert "base64-image-data" in called_messages[0]["content"][1]["image_url"]["url"]

    assert result.answer == "A cat"
