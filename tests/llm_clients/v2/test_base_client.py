import pytest
from unittest.mock import patch
from kavalai.llm_clients.base_client import (
    BaseLlmClient,
    ChatHistory,
    ChatMessage,
    LlmClientParameters,
)
import openai


class MockClient(BaseLlmClient):
    async def _run_chat_completions(self, chat_history, response_model, streamer):
        # This will be mocked in tests
        pass


@pytest.mark.asyncio
async def test_retry_logic():
    # Set a short timeout for the streamer to fail fast if it doesn't work
    params = LlmClientParameters(timeout_seconds=2.0)
    client = MockClient(llm_client_parameters=params)
    chat_history = ChatHistory(messages=[ChatMessage(role="user", content="test")])

    # Mock _run_chat_completions to fail once with a retriable error and then succeed
    call_count = 0

    async def side_effect(chat_history, response_model, streamer):
        nonlocal call_count
        call_count += 1
        print(f"Call count: {call_count}")
        if call_count == 1:
            raise openai.APIConnectionError(request=None)

        # On second call, succeed
        value_streamer = streamer.get_value_streamer("response")
        await value_streamer.stream_partial("Success")
        await value_streamer.stream_complete()

    with patch.object(MockClient, "_run_chat_completions", side_effect=side_effect):
        # We need to lower the retry delay for testing
        with patch("kavalai.llm_clients.with_retry.asyncio.sleep", return_value=None):
            streamer = await client.stream_chat_completions(chat_history=chat_history)

            contents = []
            async for content in streamer:
                contents.append(content)

            assert call_count == 2
            assert contents[-1].value == "Success"
            assert contents[-1].type == "complete"


@pytest.mark.asyncio
async def test_non_retriable_error():
    params = LlmClientParameters(timeout_seconds=1.0)
    client = MockClient(llm_client_parameters=params)
    chat_history = ChatHistory(messages=[ChatMessage(role="user", content="test")])

    async def side_effect(chat_history, response_model, streamer):
        raise ValueError("Non-retriable error")

    with patch.object(MockClient, "_run_chat_completions", side_effect=side_effect):
        # Should raise RuntimeError from the streamer
        with pytest.raises(RuntimeError, match="Non-retriable error"):
            streamer = await client.stream_chat_completions(chat_history=chat_history)
            async for _ in streamer:
                pass
