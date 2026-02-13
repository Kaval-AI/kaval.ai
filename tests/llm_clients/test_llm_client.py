from unittest.mock import AsyncMock, patch

import pytest
import openai
from google.genai import errors
from pydantic import BaseModel

from kavalai.agents.db import (
    ModelCallStat,
)
from kavalai.llm_clients.llm_client import (
    chat_completions,
    compute_embeddings,
    with_retry,
    get_llm_client,
)


class MockResponse(BaseModel):
    message: str


@pytest.mark.asyncio
async def test_with_retry_openai_rate_limit(monkeypatch):
    mock_func = AsyncMock()
    mock_func.side_effect = [
        openai.RateLimitError("Rate limit exceeded", response=AsyncMock(), body={}),
        MockResponse(message="success"),
    ]

    # Patch asyncio.sleep to speed up tests
    with patch("asyncio.sleep", return_value=None):
        result = await with_retry(mock_func, base_delay=0.1)

    assert result.message == "success"
    assert mock_func.call_count == 2


@pytest.mark.asyncio
async def test_with_retry_openai_internal_server_error():
    mock_func = AsyncMock()
    mock_func.side_effect = [
        openai.InternalServerError(
            "Internal server error", response=AsyncMock(), body={}
        ),
        MockResponse(message="success"),
    ]

    with patch("asyncio.sleep", return_value=None):
        result = await with_retry(mock_func, base_delay=0.1)

    assert result.message == "success"
    assert mock_func.call_count == 2


@pytest.mark.asyncio
async def test_with_retry_gemini_rate_limit(monkeypatch):
    mock_func = AsyncMock()
    mock_func.side_effect = [
        errors.ClientError(429, {"error": {"message": "rate limit"}}),
        MockResponse(message="success"),
    ]

    with patch("asyncio.sleep", return_value=None):
        result = await with_retry(mock_func, base_delay=0.1)

    assert result.message == "success"
    assert mock_func.call_count == 2


@pytest.mark.asyncio
async def test_with_retry_gemini_rate_limit_max_retries():
    mock_func = AsyncMock()
    mock_func.side_effect = errors.ClientError(
        429, {"error": {"message": "rate limit"}}
    )

    with patch("asyncio.sleep", return_value=None):
        with pytest.raises(errors.ClientError, match="429"):
            await with_retry(mock_func, max_retries=2, base_delay=0.1)

    assert mock_func.call_count == 3


@pytest.mark.asyncio
async def test_with_retry_non_retriable_error():
    mock_func = AsyncMock()
    mock_func.side_effect = ValueError("Fatal error")

    with pytest.raises(ValueError, match="Fatal error"):
        await with_retry(mock_func)

    assert mock_func.call_count == 1


@pytest.mark.asyncio
async def test_with_retry_openai_max_retries():
    mock_func = AsyncMock()
    mock_func.side_effect = openai.RateLimitError(
        "Rate limit exceeded", response=AsyncMock(), body={}
    )

    with patch("asyncio.sleep", return_value=None):
        with pytest.raises(openai.RateLimitError, match="Rate limit exceeded"):
            await with_retry(mock_func, max_retries=2, base_delay=0.1)

    assert mock_func.call_count == 3


@pytest.mark.asyncio
async def test_with_retry_gemini_server_error_max_retries():
    mock_func = AsyncMock()
    mock_func.side_effect = errors.ServerError(
        500, {"error": {"message": "server error"}}
    )

    with patch("asyncio.sleep", return_value=None):
        with pytest.raises(errors.ServerError, match="500"):
            await with_retry(mock_func, max_retries=2, base_delay=0.1)

    assert mock_func.call_count == 3


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
    mock_client.chat_completions.return_value = (mock_content, mock_stats)

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
    mock_client.chat_completions.side_effect = Exception("API Error")

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
    mock_client.chat_completions.side_effect = [
        openai.RateLimitError("Rate limit exceeded", response=AsyncMock(), body={}),
        (
            MockResponse(message="success"),
            ModelCallStat(
                call_type="llm",
                model="openai/gpt-4o",
                response_code=200,
                duration_seconds=0.1,
                total_tokens=10,
            ),
        ),
    ]

    monkeypatch.setattr(
        "kavalai.llm_clients.common.get_llm_client", lambda _: mock_client
    )

    with patch("asyncio.sleep", return_value=None):
        response, stats = await chat_completions(
            model="openai/gpt-4o",
            response_model=MockResponse,
            messages=[{"role": "user", "content": "hi"}],
        )

    assert response.message == "success"
    assert mock_client.chat_completions.call_count == 2


@pytest.mark.asyncio
async def test_compute_embeddings_retry(agents_db, monkeypatch):
    mock_client = AsyncMock()
    mock_client.compute_embeddings.side_effect = [
        openai.RateLimitError("Rate limit exceeded", response=AsyncMock(), body={}),
        (
            [[0.1, 0.2]],
            ModelCallStat(
                call_type="embedding",
                model="openai/text-embedding-3-small",
                response_code=200,
                duration_seconds=0.1,
                total_tokens=5,
            ),
        ),
    ]

    monkeypatch.setattr(
        "kavalai.llm_clients.common.get_llm_client", lambda _: mock_client
    )

    with patch("asyncio.sleep", return_value=None):
        embeddings, stats = await compute_embeddings(
            model="openai/text-embedding-3-small",
            texts=["hi"],
        )

    assert embeddings == [[0.1, 0.2]]
    assert mock_client.compute_embeddings.call_count == 2


@pytest.mark.asyncio
async def test_get_llm_client_invalid():
    with pytest.raises(ValueError, match="Invalid provider"):
        get_llm_client("invalid/model")


@pytest.mark.asyncio
async def test_get_llm_client_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setenv("KAVALAI_LLM_TIMEOUT", "45.0")
    with patch("kavalai.llm_clients.openai.AsyncOpenAI") as mock_openai:
        get_llm_client("openai/gpt-4")
        mock_openai.assert_called_once()
        assert mock_openai.call_args.kwargs["timeout"] == 45.0


@pytest.mark.asyncio
async def test_get_llm_client_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setenv("KAVALAI_LLM_TIMEOUT", "15.0")
    from kavalai.llm_clients.gemini_client import GeminiClient

    with patch("google.genai.Client") as mock_gemini:
        client = get_llm_client("gemini/gemini-pro")
        assert isinstance(client, GeminiClient)
        mock_gemini.assert_called_once()
        assert mock_gemini.call_args.kwargs["http_options"]["timeout"] == 15.0


@pytest.mark.asyncio
async def test_get_llm_client_default_timeout(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.delenv("KAVALAI_LLM_TIMEOUT", raising=False)
    with patch("kavalai.llm_clients.openai.AsyncOpenAI") as mock_openai:
        get_llm_client("openai/gpt-4")
        assert mock_openai.call_args.kwargs["timeout"] == 30.0


@pytest.mark.asyncio
async def test_with_retry_gemini_other_client_error_retried():
    mock_func = AsyncMock()
    # 400 Bad Request should now be retried according to updated instructions
    mock_func.side_effect = [
        errors.ClientError(400, {"error": {"message": "bad request"}}),
        MockResponse(message="success"),
    ]

    with patch("asyncio.sleep", return_value=None):
        result = await with_retry(mock_func, base_delay=0.1)

    assert result.message == "success"
    assert mock_func.call_count == 2
