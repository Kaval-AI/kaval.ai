import os
import pytest
from pydantic import BaseModel

from kavalai.llm_clients.v2.openai_client import OpenAIClient
from kavalai.llm_clients.base_client import (
    ChatHistory,
    ChatMessage,
    LlmClientParameters,
)


class SimpleResponse(BaseModel):
    answer: str


@pytest.fixture
def openaiclient():
    return OpenAIClient(
        model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY", "fake")
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_openaichat_completions(openaiclient):
    chat_history = ChatHistory(
        messages=[ChatMessage(role="user", content="Say 'Hello'")]
    )

    streamer = await openaiclient.chat_completions(chat_history=chat_history)

    contents = []
    async for content in streamer:
        contents.append(content)

    assert len(contents) >= 2
    assert any(c.type == "partial" for c in contents)
    assert contents[-1].type == "complete"
    assert "Hello" in contents[-1].value


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_openaistructured_output(openaiclient):
    chat_history = ChatHistory(
        messages=[
            ChatMessage(
                role="user", content="What is the capital of France? Respond in JSON."
            )
        ]
    )

    streamer = await openaiclient.chat_completions(
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
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_openai_parameters():
    params = LlmClientParameters(
        temperature=0.0, top_p=1.0, service_tier="priority", timeout_seconds=45.0
    )
    client = OpenAIClient(model="gpt-4o-mini", llm_client_parameters=params)

    assert client.client.timeout == 45.0

    chat_history = ChatHistory(messages=[ChatMessage(role="user", content="Say 'Hi'")])

    streamer = await client.chat_completions(chat_history=chat_history)
    async for content in streamer:
        if content.type == "complete":
            assert "Hi" in content.value
