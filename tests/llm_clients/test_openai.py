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
async def test_openai_chat_completion_retry_on_length_limit():
    client = OpenAIClient(api_key="fake-key")

    # Mock the openai client's parse method
    mock_parse = AsyncMock()

    # Mock behavior: raise LengthFinishReasonError twice, then succeed
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.parsed = SimpleResponse(
        answer="success", confidence=1.0
    )
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 30
    mock_response.model_dump.return_value = {"fake": "response"}

    # We need to simulate the openai.LengthFinishReasonError.
    # It requires a 'completion' argument in its __init__
    fake_completion = AsyncMock()
    error = LengthFinishReasonError(completion=fake_completion)

    mock_parse.side_effect = [error, error, mock_response]

    with patch.object(client.client.beta.chat.completions, "parse", mock_parse):
        content, stats = await client.chat_completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            response_model=SimpleResponse,
        )

        assert content.answer == "success"
        assert mock_parse.call_count == 3


@pytest.mark.asyncio
async def test_openai_chat_completion_fails_after_3_attempts():
    client = OpenAIClient(api_key="fake-key")
    mock_parse = AsyncMock()

    fake_completion = AsyncMock()
    error = LengthFinishReasonError(completion=fake_completion)

    # Fail 3 times (or more)
    mock_parse.side_effect = [error, error, error, error]

    with patch.object(client.client.beta.chat.completions, "parse", mock_parse):
        with pytest.raises(LengthFinishReasonError):
            await client.chat_completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                response_model=SimpleResponse,
            )
        assert mock_parse.call_count == 3


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
