import asyncio
import os
import pytest
from pydantic import BaseModel
from unittest.mock import AsyncMock, patch
from kavalai.llm_clients.gemini_client import GeminiClient
from kavalai.llm_clients.common import Streamer


class SimpleResponse(BaseModel):
    answer: str
    confidence: float


@pytest.mark.asyncio
async def test_gemini_structured_output():
    client = GeminiClient(api_key="fake-key")

    mock_response = AsyncMock()
    mock_response.text = '{"answer": "4", "confidence": 1.0}'
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 5
    mock_response.usage_metadata.total_token_count = 15

    # Mocking async iterator
    async def mock_stream(*args, **kwargs):
        yield mock_response

    with patch.object(
        client.client.aio.models, "generate_content_stream", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = mock_stream()

        messages = [
            {
                "role": "user",
                "content": "What is 2+2? Answer in JSON format with 'answer' and 'confidence' fields.",
            }
        ]
        content, stats = await client.chat_completions(
            model="gemini-2.0-flash", messages=messages, response_model=SimpleResponse
        )

        assert isinstance(content, SimpleResponse)
        assert "4" in content.answer
        assert content.confidence >= 0.0
        assert stats.total_tokens == 15
        assert stats.model == "gemini/gemini-2.0-flash"


@pytest.mark.asyncio
async def test_gemini_streaming():
    client = GeminiClient(api_key="fake-key")

    mock_response1 = AsyncMock()
    mock_response1.text = '{"answer":'
    mock_response1.usage_metadata = None

    mock_response2 = AsyncMock()
    mock_response2.text = ' "4", "confidence": 1.0}'
    mock_response2.usage_metadata.prompt_token_count = 10
    mock_response2.usage_metadata.candidates_token_count = 5
    mock_response2.usage_metadata.total_token_count = 15

    async def mock_stream(*args, **kwargs):
        yield mock_response1
        yield mock_response2

    with patch.object(
        client.client.aio.models, "generate_content_stream", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = mock_stream()

        messages = [{"role": "user", "content": "What is 2+2?"}]
        queue = asyncio.Queue()
        streamer = Streamer(name="test", queue=queue)
        content, stats = await client.chat_completions(
            model="gemini-2.0-flash",
            messages=messages,
            response_model=SimpleResponse,
            streamer=streamer,
        )

        assert isinstance(content, SimpleResponse)
        assert content.answer == "4"

        stream_content = ""
        while not queue.empty():
            stream_content += await queue.get() + "\n"

        assert '"partial"' in stream_content
        assert '"complete"' in stream_content
        assert "4" in stream_content
        assert stats.total_tokens == 15


@pytest.mark.asyncio
@pytest.mark.skipif(
    not (
        os.getenv("KAVALAI_RUN_INTEGRATION") == "true" and os.getenv("GEMINI_API_KEY")
    ),
    reason="Integration test disabled (set KAVALAI_RUN_INTEGRATION=true and provide GEMINI_API_KEY)",
)
async def test_gemini_structured_output_integration():
    api_key = os.getenv("GEMINI_API_KEY")
    client = GeminiClient(api_key=api_key)

    messages = [
        {
            "role": "user",
            "content": "What is 2+2? Answer in JSON format with 'answer' and 'confidence' fields.",
        }
    ]
    content, stats = await client.chat_completions(
        model="gemini-2.0-flash", messages=messages, response_model=SimpleResponse
    )

    assert isinstance(content, SimpleResponse)
    assert "4" in content.answer
    assert content.confidence >= 0.0
