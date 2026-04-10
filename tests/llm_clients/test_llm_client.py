from unittest.mock import AsyncMock, patch

import pytest
import openai
from google.genai import errors
from pydantic import BaseModel

from kavalai.agents.db import (
    ModelCallStat,
)
from kavalai.llm_clients.llm_client import (
    with_retry,
    LLMClient,
)
from kavalai.llm_clients.common import Streamer
import asyncio


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
    mock_underlying_client = AsyncMock()

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
    mock_underlying_client.chat_completions.return_value = (mock_content, mock_stats)

    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    with patch(
        "kavalai.llm_clients.llm_client.LLMClient._get_underlying_client",
        return_value=mock_underlying_client,
    ):
        client = LLMClient(model="openai/gpt-4o")

        messages = [{"role": "user", "content": "hi"}]
        response, stats = await client.chat_completions(
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
async def test_chat_completions_streaming(agents_db, monkeypatch):
    # Mock client
    mock_underlying_client = AsyncMock()

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
    mock_underlying_client.chat_completions.return_value = (mock_content, mock_stats)

    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    with patch(
        "kavalai.llm_clients.llm_client.LLMClient._get_underlying_client",
        return_value=mock_underlying_client,
    ):
        client = LLMClient(model="openai/gpt-4o")

        queue = asyncio.Queue()
        streamer = Streamer(name="test", queue=queue)

        messages = [{"role": "user", "content": "hi"}]

        response, stats = await client.chat_completions(
            response_model=MockResponse,
            messages=messages,
            streamer=streamer,
        )

    assert response.message == "hello"
    # Verify streamer was passed to mock_underlying_client.chat_completions
    mock_underlying_client.chat_completions.assert_called_once()
    args, kwargs = mock_underlying_client.chat_completions.call_args
    assert kwargs["streamer"] == streamer
    assert stats.request_data["requests"][0]["arguments"]["streamer"] == str(streamer)


@pytest.mark.asyncio
async def test_chat_completions_error(agents_db, monkeypatch):
    mock_underlying_client = AsyncMock()
    mock_underlying_client.chat_completions.side_effect = Exception("API Error")

    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    with patch(
        "kavalai.llm_clients.llm_client.LLMClient._get_underlying_client",
        return_value=mock_underlying_client,
    ):
        client = LLMClient(model="openai/gpt-4o")

        with pytest.raises(Exception, match="API Error"):
            await client.chat_completions(
                response_model=MockResponse,
                messages=[{"role": "user", "content": "hi"}],
            )


@pytest.mark.asyncio
async def test_chat_completions_retry(agents_db, monkeypatch):
    mock_underlying_client = AsyncMock()
    mock_underlying_client.chat_completions.side_effect = [
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

    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    with patch(
        "kavalai.llm_clients.llm_client.LLMClient._get_underlying_client",
        return_value=mock_underlying_client,
    ):
        client = LLMClient(model="openai/gpt-4o")

        with patch("asyncio.sleep", return_value=None):
            response, stats = await client.chat_completions(
                response_model=MockResponse,
                messages=[{"role": "user", "content": "hi"}],
            )

    assert response.message == "success"
    assert mock_underlying_client.chat_completions.call_count == 2


@pytest.mark.asyncio
async def test_compute_embeddings_retry(agents_db, monkeypatch):
    mock_underlying_client = AsyncMock()
    mock_underlying_client.compute_embeddings.side_effect = [
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

    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    with patch(
        "kavalai.llm_clients.llm_client.LLMClient._get_underlying_client",
        return_value=mock_underlying_client,
    ):
        client = LLMClient(model="openai/text-embedding-3-small")

        with patch("asyncio.sleep", return_value=None):
            embeddings, stats = await client.compute_embeddings(
                texts=["hi"],
            )

    assert embeddings == [[0.1, 0.2]]
    assert mock_underlying_client.compute_embeddings.call_count == 2


@pytest.mark.asyncio
async def test_llm_client_invalid():
    with pytest.raises(ValueError, match="not enough values to unpack"):
        LLMClient("invalid-model")


@pytest.mark.asyncio
async def test_llm_client_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setenv("KAVALAI_LLM_TIMEOUT", "45.0")
    with patch("kavalai.llm_clients.openai_client.AsyncOpenAI") as mock_openai:
        LLMClient("openai/gpt-4")
        mock_openai.assert_called_once()
        assert mock_openai.call_args.kwargs["timeout"] == 45.0


@pytest.mark.asyncio
async def test_llm_client_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setenv("KAVALAI_LLM_TIMEOUT", "15.0")
    from kavalai.llm_clients.gemini_client import GeminiClient

    with patch("google.genai.Client") as mock_genai:
        client = LLMClient("gemini/gemini-pro")
        assert isinstance(client.client, GeminiClient)
        assert client.client.timeout == 15.0
        mock_genai.assert_called_once_with(api_key="fake")


@pytest.mark.asyncio
async def test_llm_client_default_timeout(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.delenv("KAVALAI_LLM_TIMEOUT", raising=False)
    with patch("kavalai.llm_clients.openai_client.AsyncOpenAI") as mock_openai:
        LLMClient("openai/gpt-4")
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


@pytest.mark.asyncio
async def test_compute_embeddings_with_normalizer(monkeypatch):
    from kavalai.normalizer import Normalizer

    mock_underlying_client = AsyncMock()
    mock_underlying_client.compute_embeddings.return_value = (
        [[0.1, 0.2]],
        ModelCallStat(
            call_type="embedding",
            model="openai/text-embedding-3-small",
            duration_seconds=0.1,
        ),
    )

    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    with patch(
        "kavalai.llm_clients.llm_client.LLMClient._get_underlying_client",
        return_value=mock_underlying_client,
    ):
        client = LLMClient(model="openai/text-embedding-3-small")

        normalizer = Normalizer(l1=True)
        embeddings, stats = await client.compute_embeddings(
            texts=["hi"],
            normalize=True,
            normalizer=normalizer,
        )

    # Check that normalizer was passed to client.compute_embeddings
    mock_underlying_client.compute_embeddings.assert_called_once()
    assert (
        mock_underlying_client.compute_embeddings.call_args.kwargs["normalizer"]
        is normalizer
    )
    assert (
        mock_underlying_client.compute_embeddings.call_args.kwargs["normalize"] is True
    )


@pytest.mark.asyncio
async def test_compute_embeddings_with_model_in_kwargs(monkeypatch):
    """Test that model in kwargs doesn't cause error in compute_embeddings."""
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    client = LLMClient("ollama/llama3.2:1b")

    async def mock_compute_embeddings(*args, **kwargs):
        from kavalai.llm_clients.common import create_model_call_stat

        return [[0.1]], create_model_call_stat(
            call_type="embedding",
            model="ollama/llama3.2:1b",
            duration_sections=0.1,
            batch_size=1,
            total_tokens=1,
            cost=None,
        )

    with patch.object(
        client.client, "compute_embeddings", side_effect=mock_compute_embeddings
    ):
        # Passing 'model' in kwargs should not fail with 'multiple values for keyword argument'
        embeddings, stats = await client.compute_embeddings(
            texts=["test"], model="some-other-model"
        )
        assert embeddings == [[0.1]]


@pytest.mark.asyncio
async def test_chat_completions_with_model_in_kwargs(monkeypatch):
    """Test that model in kwargs doesn't cause error in chat_completions."""
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    client = LLMClient("ollama/llama3.2:1b")

    async def mock_chat_completions(*args, **kwargs):
        from kavalai.llm_clients.common import create_model_call_stat

        return "hello", create_model_call_stat(
            call_type="llm",
            model="ollama/llama3.2:1b",
            duration_sections=0.1,
            prompt_tokens=1,
            completion_tokens=1,
            cost=None,
        )

    with patch.object(
        client.client, "chat_completions", side_effect=mock_chat_completions
    ):
        # Passing 'model' in kwargs should not fail with 'multiple values for keyword argument'
        content, stats = await client.chat_completions(
            messages=[{"role": "user", "content": "hi"}], model="some-other-model"
        )
        assert content == "hello"
