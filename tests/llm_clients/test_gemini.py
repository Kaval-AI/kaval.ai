import os
import json
import pytest
from pydantic import BaseModel

from kavalai.llm_clients.gemini_client import GeminiClient
from kavalai.llm_clients.base_client import (
    ChatHistory,
    ChatMessage,
    LlmClientParameters,
)


class SimpleResponse(BaseModel):
    answer: str


# Supported Gemini models usable with the generateContent API, as of June 2026.
# Sourced from the Gemini API models documentation
# (https://ai.google.dev/gemini-api/docs/models). Gemini 1.5 and 2.0 are retired.
SUPPORTED_GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
@pytest.mark.parametrize("model", SUPPORTED_GEMINI_MODELS)
async def test_gemini_supported_models_integration(model):
    """Hit the real generateContent API for every supported model with params set.

    Sends temperature/top_p (universally supported) plus reasoning_effort, which
    the client maps to a thinking config; a real call must succeed for every model.
    """
    params = LlmClientParameters(
        temperature=0.0, top_p=1.0, reasoning_effort="low", timeout_seconds=60.0
    )
    client = GeminiClient(model=model, llm_client_parameters=params)
    chat_history = ChatHistory(
        messages=[ChatMessage(role="user", content="Reply with the single word: Hello")]
    )

    streamer = await client.stream_chat_completions(chat_history=chat_history)

    contents = []
    async for content in streamer:
        contents.append(content)

    assert contents[-1].type == "complete"
    assert contents[-1].value.strip() != ""


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
@pytest.mark.parametrize("model", SUPPORTED_GEMINI_MODELS)
async def test_gemini_supported_models_structured_output(model):
    """Structured output (response_schema) works against every supported model."""
    params = LlmClientParameters(timeout_seconds=60.0)
    client = GeminiClient(model=model, llm_client_parameters=params)
    chat_history = ChatHistory(
        messages=[
            ChatMessage(
                role="user",
                content="What is the capital of France? Respond in JSON.",
            )
        ]
    )

    streamer = await client.stream_chat_completions(
        chat_history=chat_history, response_model=SimpleResponse
    )

    contents = []
    async for content in streamer:
        contents.append(content)

    assert contents[-1].type == "complete"
    data = json.loads(contents[-1].value)
    assert "Paris" in data["answer"]
