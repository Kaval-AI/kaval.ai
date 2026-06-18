import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from kavalai.llm_clients.openai_client import OpenAIClient
from kavalai.llm_clients.gemini_client import GeminiClient
from kavalai.llm_clients.ollama_client import OllamaClient
from kavalai.llm_clients.base_client import (
    ChatHistory,
    ChatMessage,
    ModelStatsReceiver,
    ModelCallStat,
)


class MockReceiver(ModelStatsReceiver):
    def __init__(self):
        self.received_stats = []

    def receive_model_stats(self, stats: ModelCallStat):
        self.received_stats.append(stats)


@pytest.mark.asyncio
async def test_openai_stats_collection():
    receiver = MockReceiver()
    client = OpenAIClient(
        model="gpt-4o", api_key="test-key", model_stats_receiver=receiver
    )
    chat_history = ChatHistory(messages=[ChatMessage(role="user", content="Hi")])

    mock_stream = AsyncMock()

    # Events to yield
    from openai.types.responses import ResponseTextDeltaEvent, ResponseCompletedEvent

    class MockUsage:
        def __init__(self):
            self.input_tokens = 10
            self.output_tokens = 20

    class MockResponse:
        def __init__(self):
            self.usage = MockUsage()

    # Use MagicMock for events to avoid Pydantic validation issues
    mock_event1 = MagicMock(spec=ResponseTextDeltaEvent)
    mock_event1.delta = "Hello"
    mock_event1.type = "response.output_text.delta"

    mock_event2 = MagicMock(spec=ResponseCompletedEvent)
    mock_event2.type = "response.completed"

    mock_event2.response = MockResponse()

    events = [mock_event1, mock_event2]

    mock_stream.__aiter__.return_value = events
    mock_stream.__aenter__.return_value = mock_stream

    with patch.object(client.client.responses, "stream", return_value=mock_stream):
        # Use stream_chat_completions to get the streamer
        streamer = await client.stream_chat_completions(chat_history=chat_history)
        async for _ in streamer:
            pass

    assert len(receiver.received_stats) == 1
    stats = receiver.received_stats[0]
    assert stats.call_type == "llm"
    assert stats.model == "openai/gpt-4o"
    assert stats.prompt_tokens == 10
    assert stats.completion_tokens == 20
    assert stats.total_tokens == 30
    assert stats.response_data == "Hello"
    assert stats.request_data is not None


@pytest.mark.asyncio
async def test_gemini_stats_collection():
    receiver = MockReceiver()
    client = GeminiClient(
        model="gemini-1.5-flash", api_key="test-key", model_stats_receiver=receiver
    )
    chat_history = ChatHistory(messages=[ChatMessage(role="user", content="Hi")])

    mock_generate_content_stream = AsyncMock()

    class MockPart:
        def __init__(self, text):
            self.text = text
            self.thought = False

    class MockContent:
        def __init__(self, text):
            self.parts = [MockPart(text)]

    class MockCandidate:
        def __init__(self, text):
            self.content = MockContent(text)

    class MockUsageMetadata:
        def __init__(self):
            self.prompt_token_count = 15
            self.candidates_token_count = 25

    class MockChunk:
        def __init__(self, text=None, usage=None):
            self.candidates = [MockCandidate(text)] if text else []
            self.usage_metadata = usage

    chunks = [MockChunk(text="Hello"), MockChunk(usage=MockUsageMetadata())]

    mock_generate_content_stream.return_value = AsyncMock()
    mock_generate_content_stream.return_value.__aiter__.return_value = chunks

    with patch.object(
        client.client.aio.models,
        "generate_content_stream",
        return_value=mock_generate_content_stream.return_value,
    ):
        # Use stream_chat_completions to get the streamer
        streamer = await client.stream_chat_completions(chat_history=chat_history)
        async for _ in streamer:
            pass

    assert len(receiver.received_stats) == 1
    stats = receiver.received_stats[0]
    assert stats.call_type == "llm"
    assert stats.model == "gemini/gemini-1.5-flash"
    assert stats.prompt_tokens == 15
    assert stats.completion_tokens == 25
    assert stats.total_tokens == 40
    assert stats.response_data == "Hello"


@pytest.mark.asyncio
async def test_ollama_stats_collection():
    receiver = MockReceiver()
    client = OllamaClient(model="llama3", model_stats_receiver=receiver)
    chat_history = ChatHistory(messages=[ChatMessage(role="user", content="Hi")])

    mock_chat = AsyncMock()

    chunks = [
        {"message": {"role": "assistant", "content": "Hello"}, "done": False},
        {"done": True, "prompt_eval_count": 5, "eval_count": 10},
    ]

    mock_chat.return_value = AsyncMock()
    mock_chat.return_value.__aiter__.return_value = chunks

    with patch.object(client.client, "chat", return_value=mock_chat.return_value):
        # Use stream_chat_completions to get the streamer
        streamer = await client.stream_chat_completions(chat_history=chat_history)
        async for _ in streamer:
            pass

    assert len(receiver.received_stats) == 1
    stats = receiver.received_stats[0]
    assert stats.call_type == "llm"
    assert stats.model == "ollama/llama3"
    assert stats.prompt_tokens == 5
    assert stats.completion_tokens == 10
    assert stats.total_tokens == 15
    assert stats.response_data == "Hello"
