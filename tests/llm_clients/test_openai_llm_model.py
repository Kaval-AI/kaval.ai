import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from pydantic import BaseModel

from kavalai.llm_clients.common import Streamer, StreamContent
from kavalai.llm_clients.providers.openai_llm_model import (
    OpenAILlmModel,
    OpenAILlmModelConfig,
)


class SimpleResponse(BaseModel):
    answer: str
    confidence: float


@pytest.fixture
def openai_model():
    config = OpenAILlmModelConfig(
        model_name="gpt-4o-mini",
        api_key="fake",
    )
    return OpenAILlmModel(config)


@pytest.mark.asyncio
async def test_openai_llm_model_chat_completions_unit(openai_model):
    """Unit test for chat_completions with mocking."""
    from openai.types.responses import (
        ResponseTextDeltaEvent,
        ResponseCompletedEvent,
    )

    messages = [{"role": "user", "content": "Say 'Hello'"}]

    # Mock events
    mock_text_event = MagicMock(spec=ResponseTextDeltaEvent)
    mock_text_event.delta = "Hello"

    mock_completed_event = MagicMock(spec=ResponseCompletedEvent)
    # Properly mock the nested structure
    mock_completed_event.response = MagicMock()
    mock_completed_event.response.usage = MagicMock()
    mock_completed_event.response.usage.input_tokens = 10
    mock_completed_event.response.usage.output_tokens = 5

    # Mock stream context manager
    mock_stream = MagicMock()
    mock_stream.__aenter__.return_value = AsyncMock()
    mock_stream.__aenter__.return_value.__aiter__.return_value = [
        mock_text_event,
        mock_completed_event,
    ]

    with patch(
        "openai.resources.responses.AsyncResponses.stream", return_value=mock_stream
    ) as mock_stream_call:
        result, stats = await openai_model.chat_completions(messages=messages)

        assert result == "Hello"
        assert stats.prompt_tokens == 10
        assert stats.completion_tokens == 5
        assert stats.model == "openai/gpt-4o-mini"

        mock_stream_call.assert_called_once()
        _, kwargs = mock_stream_call.call_args
        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["input"] == messages


@pytest.mark.asyncio
async def test_openai_llm_model_chat_completions_formatted_unit(openai_model):
    """Unit test for formatted chat_completions with mocking."""
    from openai.types.responses import (
        ResponseTextDeltaEvent,
        ResponseCompletedEvent,
    )

    messages = [{"role": "user", "content": "Give me a response"}]
    response_json = '{"answer": "Paris", "confidence": 0.9}'

    # Mock events
    mock_text_event = MagicMock(spec=ResponseTextDeltaEvent)
    mock_text_event.delta = response_json

    mock_completed_event = MagicMock(spec=ResponseCompletedEvent)
    # Properly mock the nested structure
    mock_completed_event.response = MagicMock()
    mock_completed_event.response.usage = MagicMock()
    mock_completed_event.response.usage.input_tokens = 15
    mock_completed_event.response.usage.output_tokens = 20

    # Mock stream context manager
    mock_stream = MagicMock()
    mock_stream.__aenter__.return_value = AsyncMock()
    mock_stream.__aenter__.return_value.__aiter__.return_value = [
        mock_text_event,
        mock_completed_event,
    ]

    with patch(
        "openai.resources.responses.AsyncResponses.stream", return_value=mock_stream
    ) as mock_stream_call:
        result, stats = await openai_model.chat_completions(
            messages=messages, response_model=SimpleResponse
        )

        assert isinstance(result, SimpleResponse)
        assert result.answer == "Paris"
        assert result.confidence == 0.9
        assert stats.prompt_tokens == 15
        assert stats.completion_tokens == 20

        mock_stream_call.assert_called_once()
        _, kwargs = mock_stream_call.call_args
        assert kwargs["text_format"] == SimpleResponse


@pytest.mark.asyncio
async def test_openai_llm_model_streamer_unit(openai_model):
    """Unit test for streamer with mocking."""
    from openai.types.responses import (
        ResponseTextDeltaEvent,
        ResponseCompletedEvent,
    )

    messages = [{"role": "user", "content": "Say 'Hi'"}]
    queue = asyncio.Queue()
    streamer = Streamer("test_streamer", queue)

    # Mock events
    mock_text_event1 = MagicMock(spec=ResponseTextDeltaEvent)
    mock_text_event1.delta = "H"
    mock_text_event2 = MagicMock(spec=ResponseTextDeltaEvent)
    mock_text_event2.delta = "i"

    mock_completed_event = MagicMock(spec=ResponseCompletedEvent)
    # Properly mock the nested structure
    mock_completed_event.response = MagicMock()
    mock_completed_event.response.usage = MagicMock()
    mock_completed_event.response.usage.input_tokens = 5
    mock_completed_event.response.usage.output_tokens = 2

    # Mock stream context manager
    mock_stream = MagicMock()
    mock_stream.__aenter__.return_value = AsyncMock()
    mock_stream.__aenter__.return_value.__aiter__.return_value = [
        mock_text_event1,
        mock_text_event2,
        mock_completed_event,
    ]

    with patch(
        "openai.resources.responses.AsyncResponses.stream", return_value=mock_stream
    ):
        result, _ = await openai_model.chat_completions(
            messages=messages, streamer=streamer
        )

        assert result == "Hi"

        partials = []
        final_streamed = None
        while not queue.empty():
            content_json = await queue.get()
            content = StreamContent.model_validate_json(content_json)
            if content.type == "partial":
                partials.append(content.value)
            elif content.type == "complete":
                final_streamed = content.value

        assert partials == ["H", "Hi"]
        assert final_streamed == "Hi"


@pytest.mark.asyncio
async def test_openai_llm_model_stream_delta_unit(openai_model):
    """Unit test for stream_delta with mocking."""
    from openai.types.responses import (
        ResponseTextDeltaEvent,
        ResponseCompletedEvent,
    )

    messages = [{"role": "user", "content": "Say 'Hi'"}]
    queue = asyncio.Queue()
    streamer = Streamer("test_streamer_delta", queue)

    # Mock events
    mock_text_event1 = MagicMock(spec=ResponseTextDeltaEvent)
    mock_text_event1.delta = "H"
    mock_text_event2 = MagicMock(spec=ResponseTextDeltaEvent)
    mock_text_event2.delta = "i"

    mock_completed_event = MagicMock(spec=ResponseCompletedEvent)
    mock_completed_event.response = MagicMock()
    mock_completed_event.response.usage = MagicMock()
    mock_completed_event.response.usage.input_tokens = 5
    mock_completed_event.response.usage.output_tokens = 2

    # Mock stream context manager
    mock_stream = MagicMock()
    mock_stream.__aenter__.return_value = AsyncMock()
    mock_stream.__aenter__.return_value.__aiter__.return_value = [
        mock_text_event1,
        mock_text_event2,
        mock_completed_event,
    ]

    with patch(
        "openai.resources.responses.AsyncResponses.stream", return_value=mock_stream
    ):
        result, _ = await openai_model.chat_completions(
            messages=messages, streamer=streamer, stream_delta=True
        )

        assert result == "Hi"

        partials = []
        while not queue.empty():
            content_json = await queue.get()
            content = StreamContent.model_validate_json(content_json)
            if content.type == "partial":
                partials.append(content.value)

        assert partials == ["H", "i"]
