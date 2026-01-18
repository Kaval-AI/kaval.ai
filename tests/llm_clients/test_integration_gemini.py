import os
import pytest
from pydantic import BaseModel
from kavalai.llm_clients.gemini import GeminiClient


class SimpleResponse(BaseModel):
    answer: str
    confidence: float


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
async def test_gemini_chat_completion():
    api_key = os.getenv("GEMINI_API_KEY")
    client = GeminiClient(api_key=api_key)

    messages = [{"role": "user", "content": "Say hello!"}]
    response = await client.chat_completion(model="gemini-2.0-flash", messages=messages)

    assert "content" in response
    assert isinstance(response["content"], str)
    assert len(response["content"]) > 0
    assert "usage" in response
    assert "cost" in response
    assert response["model"] == "gemini-2.0-flash"


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
async def test_gemini_structured_output():
    api_key = os.getenv("GEMINI_API_KEY")
    client = GeminiClient(api_key=api_key)

    messages = [
        {
            "role": "user",
            "content": "What is 2+2? Answer in JSON format with 'answer' and 'confidence' fields.",
        }
    ]
    response = await client.chat_completion(
        model="gemini-2.0-flash", messages=messages, response_model=SimpleResponse
    )

    assert isinstance(response["content"], SimpleResponse)
    assert "4" in response["content"].answer
    assert response["content"].confidence >= 0.0
