import os

import pytest
from pydantic import BaseModel

from kavalai.llm_clients.base_client import (
    ChatHistory,
    ChatMessage,
    LlmClientParameters,
)
from kavalai.llm_clients.v2.gemini_client import GeminiClient


class SimpleResponse(BaseModel):
    answer: str


@pytest.fixture
def gemini_client():
    return GeminiClient(
        model="gemini-1.5-flash", api_key=os.getenv("GEMINI_API_KEY", "fake")
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
async def test_gemini_chat_completions(gemini_client):
    chat_history = ChatHistory(
        messages=[ChatMessage(role="user", content="Say 'Hello'")]
    )

    streamer = await gemini_client.chat_completions(chat_history=chat_history)

    contents = []
    async for content in streamer:
        contents.append(content)

    assert len(contents) >= 2
    assert any(c.type == "partial" for c in contents)
    assert contents[-1].type == "complete"
    assert "Hello" in contents[-1].value


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
async def test_gemini_structured_output(gemini_client):
    chat_history = ChatHistory(
        messages=[
            ChatMessage(
                role="user", content="What is the capital of France? Respond in JSON."
            )
        ]
    )

    streamer = await gemini_client.chat_completions(
        chat_history=chat_history, response_model=SimpleResponse
    )

    contents = []
    async for content in streamer:
        contents.append(content)

    assert contents[-1].type == "complete"
    import json

    data = json.loads(contents[-1].value)
    assert "Paris" in data["answer"]


@pytest.mark.asyncio
async def test_gemini_parameters():
    params = LlmClientParameters(
        temperature=0.0, top_p=1.0, service_tier="priority", timeout_seconds=45.0
    )
    client = GeminiClient(
        model="gemini-1.5-flash", llm_client_parameters=params, api_key="fake"
    )

    assert client.timeout == 45.0
    assert client.model == "gemini-1.5-flash"


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
async def test_gemini_thinking_real(gemini_client):
    # Use a thinking model
    gemini_client.model = "gemini-2.0-flash-thinking-exp"
    gemini_client.parameters = LlmClientParameters(reasoning_effort="high")

    chat_history = ChatHistory(
        messages=[ChatMessage(role="user", content="Solve 12345 * 67890 step by step.")]
    )
    streamer = await gemini_client.chat_completions(chat_history=chat_history)

    thoughts = []
    responses = []
    async for content in streamer:
        if content.name == "thought" and content.type == "partial":
            thoughts.append(content.value)
        elif content.name == "response" and content.type == "partial":
            responses.append(content.value)

    assert len(thoughts) > 0
    assert len(responses) > 0
