import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel
from kavalai.llm_clients.gemini_client import GeminiClient
from kavalai.llm_clients.common import Streamer, StreamContent


class SimpleResponse(BaseModel):
    answer: str
    confidence: float


@pytest.fixture
def gemini_client():
    with patch("google.genai.Client"):
        client = GeminiClient(api_key="fake-key")

        # Helper for mocking streams
        class AsyncIter:
            def __init__(self, items):
                self.items = items

            async def __aiter__(self):
                for item in self.items:
                    yield item

        client._AsyncIter = AsyncIter
        return client


@pytest.mark.asyncio
async def test_gemini_chat_completions_simple(gemini_client):
    """Test simple message and string result."""
    mock_response = MagicMock()
    mock_response.text = "Hello! How can I help you today?"
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 5
    mock_response.usage_metadata.total_token_count = 15

    mock_stream = gemini_client._AsyncIter([mock_response])
    gemini_client.client.aio.models.generate_content_stream = AsyncMock(
        return_value=mock_stream
    )

    messages = [{"role": "user", "content": "Say 'Hello'"}]
    result, stats = await gemini_client.chat_completions(
        model="gemini-1.5-flash", messages=messages
    )

    assert "Hello" in result
    assert stats.prompt_tokens == 10
    assert stats.completion_tokens == 5
    assert stats.model == "gemini/gemini-1.5-flash"


@pytest.mark.asyncio
async def test_gemini_chat_completions_formatted(gemini_client):
    """Test formatted result (structured output)."""
    mock_response = MagicMock()
    mock_response.text = '{"answer": "Paris", "confidence": 0.99}'
    mock_response.usage_metadata.prompt_token_count = 15
    mock_response.usage_metadata.candidates_token_count = 10

    mock_stream = gemini_client._AsyncIter([mock_response])
    gemini_client.client.aio.models.generate_content_stream = AsyncMock(
        return_value=mock_stream
    )

    messages = [{"role": "user", "content": "What is the capital of France?"}]
    result, stats = await gemini_client.chat_completions(
        model="gemini-1.5-flash", messages=messages, response_model=SimpleResponse
    )

    assert isinstance(result, SimpleResponse)
    assert "Paris" in result.answer
    assert result.confidence == 0.99
    assert stats.prompt_tokens == 15
    assert stats.completion_tokens == 10


@pytest.mark.asyncio
async def test_gemini_chat_completions_multimodal(gemini_client):
    """Test results with text and image input."""
    mock_response = MagicMock()
    mock_response.text = "The image is black."
    mock_response.usage_metadata.prompt_token_count = 20
    mock_response.usage_metadata.candidates_token_count = 10

    mock_stream = gemini_client._AsyncIter([mock_response])
    gemini_client.client.aio.models.generate_content_stream = AsyncMock(
        return_value=mock_stream
    )

    # Small 1x1 black PNG
    dummy_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What color is this image?"},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{dummy_image_b64}",
                },
            ],
        }
    ]
    result, stats = await gemini_client.chat_completions(
        model="gemini-1.5-flash", messages=messages
    )

    assert isinstance(result, str)
    assert "black" in result
    assert stats.prompt_tokens == 20


@pytest.mark.asyncio
async def test_gemini_chat_completions_streamer(gemini_client):
    """Test streamer functionality."""
    mock_response1 = MagicMock()
    mock_response1.text = '"Hi'
    mock_response2 = MagicMock()
    mock_response2.text = ' there!"'
    mock_response2.usage_metadata.prompt_token_count = 5
    mock_response2.usage_metadata.candidates_token_count = 2

    mock_stream = gemini_client._AsyncIter([mock_response1, mock_response2])
    gemini_client.client.aio.models.generate_content_stream = AsyncMock(
        return_value=mock_stream
    )

    messages = [{"role": "user", "content": "Say 'Hi'"}]

    # Create async queue and real streamer
    queue = asyncio.Queue()
    streamer = Streamer("test_streamer", queue)

    # Make API call with streamer
    result, stats = await gemini_client.chat_completions(
        model="gemini-1.5-flash", messages=messages, streamer=streamer
    )

    # Collect all streamed content
    partials = []
    final_streamed = None

    while not queue.empty():
        content_json = await queue.get()
        content = StreamContent.model_validate_json(content_json)
        if content.type == "partial":
            partials.append(content.value)
        elif content.type == "complete":
            final_streamed = content.value

    assert result == '"Hi there!"'
    assert len(partials) > 0
    assert final_streamed == result


@pytest.mark.asyncio
async def test_gemini_generate_image(gemini_client):
    """Test image generation."""
    mock_part = MagicMock()
    mock_part.inline_data.data = b"fake-image-data"
    mock_response = MagicMock()
    mock_response.parts = [mock_part]

    gemini_client.client.aio.models.generate_content = AsyncMock(
        return_value=mock_response
    )

    prompt = "A simple red square 1x1 size."
    image_base64, stats = await gemini_client.generate_image(
        model="imagen-3.0-generate-001",
        prompt=prompt,
    )
    assert image_base64 is not None
    assert isinstance(image_base64, str)
    assert stats.call_type == "image_generation"


@pytest.mark.asyncio
async def test_gemini_compute_embeddings(gemini_client):
    """Test embedding generation."""
    mock_response = MagicMock()
    mock_response.embeddings = [
        MagicMock(values=[0.1, 0.2, 0.3]),
        MagicMock(values=[0.4, 0.5, 0.6]),
    ]
    mock_response.usage_metadata.total_token_count = 10

    gemini_client.client.aio.models.embed_content = AsyncMock(
        return_value=mock_response
    )

    texts = ["Hello world", "How are you?"]
    embeddings, stats = await gemini_client.compute_embeddings(
        model="text-embedding-004", texts=texts
    )
    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert stats.call_type == "embedding"
