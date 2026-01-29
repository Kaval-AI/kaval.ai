from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel
from sqlalchemy import select

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

    # Verify stats in DB
    stmt = select(ModelCallStat).where(ModelCallStat.call_type == "llm")
    result = await agents_db.execute(stmt)
    stat = result.scalar_one()

    assert stat.prompt_tokens == 10
    assert stat.completion_tokens == 5
    assert stat.total_tokens == 15
    assert stat.response_code == 200
    assert stat.duration_seconds >= 0
    assert float(stat.cost) == 0.0001
    assert stat.request_data["requests"][0]["arguments"]["messages"] == messages
    assert stat.response_data == {"id": "chat-123"}


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
