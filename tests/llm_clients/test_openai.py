import pytest
from typing import Any
from pydantic import BaseModel
from kavalai.llm_clients.openai_client import OpenAIClient
import asyncio

from kavalai.llm_clients.common import Streamer, StreamContent


class SimpleResponse(BaseModel):
    answer: str
    confidence: float


class MockStreamer(Streamer):
    def __init__(self):
        self.partials = []
        self.final = None

    async def stream_partial(self, content: Any):
        self.partials.append(content)

    async def stream_complete(self, content: Any):
        self.final = content


@pytest.fixture
def openai_client():
    return OpenAIClient()


@pytest.mark.asyncio
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
    assert stats.cost is not None
    assert stats.cost > 0
    assert stats.currency == "USD"


@pytest.mark.asyncio
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
async def test_openai_chat_completions_multimodal(openai_client):
    """Test results with text and image input."""
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
    result, stats = await openai_client.chat_completions(
        model="gpt-4o-mini", messages=messages
    )

    assert isinstance(result, str)
    assert len(result) > 0
    assert stats.prompt_tokens > 0
    assert stats.completion_tokens > 0


@pytest.mark.asyncio
async def test_openai_chat_completions_streamer(openai_client):
    """Test streamer functionality."""
    messages = [{"role": "user", "content": "Say 'Hi'"}]

    # Create async queue and real streamer
    queue = asyncio.Queue()
    streamer = Streamer("test_streamer", queue)

    # Make API call with streamer
    result, stats = await openai_client.chat_completions(
        model="gpt-4o-mini", messages=messages, streamer=streamer
    )

    # Collect all streamed content
    partials = []
    final_streamed = None

    while not queue.empty():
        content_json = await queue.get()
        content = StreamContent.model_validate_json(content_json)
        if isinstance(content, StreamContent):
            if content.type == "partial":
                partials.append(content.value)
            elif content.type == "complete":
                final_streamed = content.value

    assert "Hi" in result
    assert len(partials) > 0  # Verify we got partial updates
    assert final_streamed == result  # Verify final streamed matches result
