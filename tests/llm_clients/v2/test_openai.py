import os
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel

from kavalai.llm_clients.v2.openai_client import OpenAIClient
from kavalai.llm_clients.base_client import (
    ChatHistory,
    ChatMessage,
    LlmClientParameters,
)
from openai.types.responses import (
    ResponseTextDeltaEvent,
    ResponseCompletedEvent,
)


class SimpleResponse(BaseModel):
    answer: str


@pytest.fixture
def openaiclient():
    return OpenAIClient(
        model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY", "fake")
    )


@pytest.mark.asyncio
async def test_openaichat_completions(openaiclient):
    chat_history = ChatHistory(
        messages=[ChatMessage(role="user", content="Say 'Hello'")]
    )

    mock_stream = AsyncMock()

    event1 = MagicMock(spec=ResponseTextDeltaEvent)
    event1.delta = "Hello"
    event1.type = "response.output_text.delta"

    event2 = MagicMock(spec=ResponseCompletedEvent)
    event2.type = "response.done"
    event2.response = MagicMock()
    event2.response.usage = MagicMock(input_tokens=10, output_tokens=5)

    async def mock_async_iterator():
        yield event1
        yield event2

    mock_stream.__aenter__.return_value = mock_async_iterator()

    with patch("kavalai.llm_clients.v2.openai_client.AsyncOpenAI") as mock_openai:
        openaiclient.client = mock_openai.return_value
        openaiclient.client.responses.stream.return_value = mock_stream

        streamer = await openaiclient.stream_chat_completions(chat_history=chat_history)

        contents = []
        async for content in streamer:
            contents.append(content)

        assert len(contents) >= 2
        assert any(c.type == "partial" for c in contents)
        assert contents[-1].type == "complete"
        assert "Hello" in contents[-1].value


@pytest.mark.asyncio
async def test_openaistructured_output(openaiclient):
    chat_history = ChatHistory(
        messages=[
            ChatMessage(
                role="user", content="What is the capital of France? Respond in JSON."
            )
        ]
    )

    mock_stream = AsyncMock()
    # For structured output, the model returns JSON string in deltas
    event1 = MagicMock(spec=ResponseTextDeltaEvent)
    event1.delta = '{"answer": "Paris"}'
    event1.type = "response.output_text.delta"

    event2 = MagicMock(spec=ResponseCompletedEvent)
    event2.type = "response.done"
    event2.response = MagicMock()
    event2.response.usage = MagicMock(input_tokens=10, output_tokens=5)

    async def mock_async_iterator():
        yield event1
        yield event2

    mock_stream.__aenter__.return_value = mock_async_iterator()

    with patch("kavalai.llm_clients.v2.openai_client.AsyncOpenAI") as mock_openai:
        openaiclient.client = mock_openai.return_value
        openaiclient.client.responses.stream.return_value = mock_stream

        streamer = await openaiclient.stream_chat_completions(
            chat_history=chat_history, response_model=SimpleResponse
        )

        contents = []
        async for content in streamer:
            contents.append(content)

        assert contents[-1].type == "complete"
        data = json.loads(contents[-1].value)
        assert "Paris" in data["answer"]


@pytest.mark.asyncio
async def test_openai_parameters():
    params = LlmClientParameters(
        temperature=0.0, top_p=1.0, service_tier="priority", timeout_seconds=45.0
    )
    client = OpenAIClient(
        model="gpt-4o-mini", llm_client_parameters=params, api_key="fake"
    )

    assert client.client.timeout == 45.0

    chat_history = ChatHistory(messages=[ChatMessage(role="user", content="Say 'Hi'")])

    mock_stream = AsyncMock()
    event1 = MagicMock(spec=ResponseTextDeltaEvent)
    event1.delta = "Hi"
    event1.type = "response.output_text.delta"

    event2 = MagicMock(spec=ResponseCompletedEvent)
    event2.type = "response.done"
    event2.response = MagicMock()
    event2.response.usage = MagicMock(input_tokens=10, output_tokens=5)

    async def mock_async_iterator():
        yield event1
        yield event2

    mock_stream.__aenter__.return_value = mock_async_iterator()

    with patch("kavalai.llm_clients.v2.openai_client.AsyncOpenAI") as mock_openai:
        mock_openai.return_value.responses.stream.return_value = mock_stream
        # We need to re-initialize or manually set the mock because AsyncOpenAI was already called in __init__
        client.client = mock_openai.return_value
        client.client.timeout = 45.0  # Keep the timeout check valid

        streamer = await client.stream_chat_completions(chat_history=chat_history)
        async for content in streamer:
            if content.type == "complete":
                assert "Hi" in content.value
