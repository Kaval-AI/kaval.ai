from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from kavalai.agents.db import (
    ModelCallStat,
)
from kavalai.llm_clients.common import chat_completions


class MockResponse(BaseModel):
    message: str


@pytest.mark.asyncio
async def test_chat_completions_with_stats(agents_db, monkeypatch):
    # Mock client
    mock_client = AsyncMock()

    # Mock response
    mock_content = MockResponse(message="hello")
    mock_stats = ModelCallStat(
        call_type="llm",
        model="openai/gpt-4o",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        duration_seconds=0.5,
        cost=None,
        response_data={"id": "chat-123"},
        response_code=200,
    )
    mock_client.chat_completion.return_value = (mock_content, mock_stats)

    monkeypatch.setattr(
        "kavalai.llm_clients.common.get_llm_client", lambda _: mock_client
    )

    messages = [{"role": "user", "content": "hi"}]
    response, stats = await chat_completions(
        model="openai/gpt-4o",
        response_model=MockResponse,
        messages=messages,
    )

    assert response.message == "hello"

    assert stats.completion_tokens == 5
    assert stats.total_tokens == 15
    assert stats.response_code == 200
    assert stats.duration_seconds >= 0
    assert stats.cost is None
    assert stats.request_data["requests"][0]["arguments"]["messages"] == messages
    assert stats.response_data == {"id": "chat-123"}


@pytest.mark.asyncio
async def test_chat_completions_error(agents_db, monkeypatch):
    mock_client = AsyncMock()
    mock_client.chat_completion.side_effect = Exception("API Error")

    monkeypatch.setattr(
        "kavalai.llm_clients.common.get_llm_client", lambda _: mock_client
    )

    with pytest.raises(Exception, match="API Error"):
        await chat_completions(
            model="openai/gpt-4o",
            response_model=MockResponse,
            messages=[{"role": "user", "content": "hi"}],
        )


@pytest.mark.asyncio
async def test_chat_completions_retry(agents_db, monkeypatch):
    mock_client = AsyncMock()
    mock_client.chat_completion.side_effect = Exception("Temporary Error")

    monkeypatch.setattr(
        "kavalai.llm_clients.common.get_llm_client", lambda _: mock_client
    )

    with pytest.raises(Exception, match="Temporary Error"):
        await chat_completions(
            model="openai/gpt-4o",
            response_model=MockResponse,
            messages=[{"role": "user", "content": "hi"}],
        )
