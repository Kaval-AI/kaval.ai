import os

import pytest
from pydantic import BaseModel

from unittest.mock import AsyncMock, patch
from openai import LengthFinishReasonError
from kavalai.llm_clients.openai import OpenAIClient


class SimpleResponse(BaseModel):
    answer: str
    confidence: float


@pytest.mark.asyncio
async def test_openai_chat_completion_no_retry_in_client():
    client = OpenAIClient(api_key="fake-key")

    # Mock the openai client's parse method
    mock_parse = AsyncMock()

    # We need to simulate the openai.LengthFinishReasonError.
    fake_completion = AsyncMock()
    error = LengthFinishReasonError(completion=fake_completion)

    mock_parse.side_effect = error

    with patch.object(client.client.beta.chat.completions, "parse", mock_parse):
        with pytest.raises(LengthFinishReasonError):
            await client.chat_completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                response_model=SimpleResponse,
            )
        assert mock_parse.call_count == 1


@pytest.mark.asyncio
async def test_openai_chat_completion_with_service_tier():
    client = OpenAIClient(api_key="fake-key", service_tier="priority")
    mock_parse = AsyncMock()

    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.parsed = SimpleResponse(
        answer="ok", confidence=1.0
    )
    mock_response.usage.total_tokens = 5
    mock_response.model_dump.return_value = {}

    mock_parse.return_value = mock_response

    with patch.object(client.client.beta.chat.completions, "parse", mock_parse):
        await client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            response_model=SimpleResponse,
        )
        assert mock_parse.call_args.kwargs["service_tier"] == "priority"


@pytest.mark.asyncio
async def test_openai_compute_embeddings_with_service_tier():
    client = OpenAIClient(api_key="fake-key", service_tier="priority")
    mock_create = AsyncMock()

    mock_data = AsyncMock()
    mock_data.embedding = [0.1]
    mock_response = AsyncMock()
    mock_response.data = [mock_data]
    mock_response.usage.total_tokens = 5
    mock_response.model_dump.return_value = {}

    mock_create.return_value = mock_response

    with patch.object(client.client.embeddings, "create", mock_create):
        await client.compute_embeddings(model="text-embedding-3-small", texts=["hi"])
        assert "service_tier" not in mock_create.call_args.kwargs


@pytest.mark.asyncio
async def test_openai_compute_embeddings():
    client = OpenAIClient(api_key="fake-key")
    mock_create = AsyncMock()

    mock_data = AsyncMock()
    mock_data.embedding = [0.1, 0.2]
    mock_response = AsyncMock()
    mock_response.data = [mock_data]
    mock_response.usage.total_tokens = 5
    mock_response.model_dump.return_value = {"fake": "resp"}

    mock_create.return_value = mock_response

    with patch.object(client.client.embeddings, "create", mock_create):
        embeddings, stats = await client.compute_embeddings(
            model="text-embedding-3-small", texts=["hi"], normalize=True
        )

    assert embeddings == [
        [0.1 / (0.1**2 + 0.2**2) ** 0.5, 0.2 / (0.1**2 + 0.2**2) ** 0.5]
    ]
    assert stats.total_tokens == 5
    assert mock_create.call_count == 1


@pytest.mark.asyncio
async def test_openai_compute_embeddings_zero_norm():
    client = OpenAIClient(api_key="fake-key")
    mock_create = AsyncMock()

    mock_data = AsyncMock()
    mock_data.embedding = [0.0, 0.0]
    mock_response = AsyncMock()
    mock_response.data = [mock_data]
    mock_response.usage.total_tokens = 0
    mock_response.model_dump.return_value = {}

    mock_create.return_value = mock_response

    with patch.object(client.client.embeddings, "create", mock_create):
        embeddings, stats = await client.compute_embeddings(
            model="text-embedding-3-small", texts=["hi"], normalize=True
        )

    assert embeddings == [[0.0, 0.0]]


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_openai_structured_output():
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAIClient(api_key=api_key)

    messages = [
        {
            "role": "user",
            "content": "What is 2+2? Answer in JSON format with 'answer' and 'confidence' fields.",
        }
    ]
    content, stats = await client.chat_completion(
        model="gpt-4o-mini", messages=messages, response_model=SimpleResponse
    )

    assert isinstance(content, SimpleResponse)
    assert "4" in content.answer
    assert content.confidence >= 0.0
