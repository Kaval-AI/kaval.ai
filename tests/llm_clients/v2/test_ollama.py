import os
import pytest
import httpx
from pydantic import BaseModel

from kavalai.llm_clients.base_client import (
    ChatHistory,
    ChatMessage,
    LlmClientParameters,
)
from kavalai.llm_clients.v2.ollama_client import OllamaClient


class SimpleResponse(BaseModel):
    answer: str


def is_ollama_running():
    host = os.getenv("OLLAMA_HOST", "localhost:11434")
    url = f"http://{host}/api/tags" if "://" not in host else f"{host}/api/tags"
    try:
        response = httpx.get(url, timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not is_ollama_running(),
    reason="OLLAMA_HOST not set or Ollama service not reachable",
)


@pytest.fixture
def model_name():
    return "llama3.2:1b"


@pytest.fixture
def ollama_client(model_name):
    return OllamaClient(model=model_name)


@pytest.mark.asyncio
async def test_ollama_chat_completions(ollama_client):
    chat_history = ChatHistory(
        messages=[ChatMessage(role="user", content="Say 'Hello'")]
    )

    streamer = await ollama_client.chat_completions(chat_history=chat_history)

    contents = []
    async for content in streamer:
        contents.append(content)

    assert len(contents) >= 2
    assert any(c.type == "partial" for c in contents)
    assert contents[-1].type == "complete"
    assert "Hello" in contents[-1].value


@pytest.mark.asyncio
async def test_ollama_structured_output(ollama_client):
    chat_history = ChatHistory(
        messages=[
            ChatMessage(
                role="user",
                content="What is the capital of France? Respond in JSON format with field 'answer'.",
            )
        ]
    )

    streamer = await ollama_client.chat_completions(
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
async def test_ollama_parameters():
    params = LlmClientParameters(temperature=0.0, top_p=1.0, timeout_seconds=45.0)
    client = OllamaClient(model="llama3.2:1b", llm_client_parameters=params)

    assert client.timeout == 45.0
    assert client.model == "llama3.2:1b"
