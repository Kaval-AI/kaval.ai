import asyncio
import pytest
from pydantic import BaseModel
from kavalai.llm_clients.openai_client import OpenAIClient
from kavalai.llm_clients.gemini_client import GeminiClient
from kavalai.llm_clients.common import Streamer, StreamContent


class SimpleResponse(BaseModel):
    answer: str


async def _pick_gemini_model(client: GeminiClient) -> str:
    models = await client.list_models()
    for cand in models:
        name = cand.lower()
        if "flash" in name and "image" not in name and "generate" not in name:
            return cand
    return models[0] if models else "gemini-1.5-flash"


@pytest.mark.asyncio
async def test_openai_stream_delta_false():
    client = OpenAIClient()
    queue = asyncio.Queue()
    streamer = Streamer("test", queue)

    # Default is stream_delta=False
    await client.chat_completions(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'Hello'"}],
        streamer=streamer,
    )

    partials = []
    while not queue.empty():
        content = StreamContent.model_validate_json(await queue.get())
        if content.type == "partial":
            partials.append(content.value)

    # With delta=False, each partial should contain the previous one
    if len(partials) > 1:
        assert partials[1].startswith(partials[0])


@pytest.mark.asyncio
async def test_gemini_stream_delta_false():
    client = GeminiClient()
    queue = asyncio.Queue()
    streamer = Streamer("test", queue)
    model = await _pick_gemini_model(client)

    # Default is stream_delta=False
    await client.chat_completions(
        model=model,
        messages=[{"role": "user", "content": "Say 'Hello'"}],
        streamer=streamer,
    )

    partials = []
    while not queue.empty():
        content = StreamContent.model_validate_json(await queue.get())
        if content.type == "partial":
            partials.append(content.value)

    # With delta=False, each partial should contain the previous one
    if len(partials) > 1:
        assert partials[1].startswith(partials[0])


@pytest.mark.asyncio
async def test_openai_stream_delta_true():
    client = OpenAIClient()
    queue = asyncio.Queue()
    streamer = Streamer("test", queue)

    await client.chat_completions(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'Hello'"}],
        streamer=streamer,
        stream_delta=True,
    )

    partials = []
    while not queue.empty():
        content = StreamContent.model_validate_json(await queue.get())
        if content.type == "partial":
            partials.append(content.value)
    if len(partials) > 1:
        if partials[0]:
            assert not partials[1].startswith(partials[0])


@pytest.mark.asyncio
async def test_gemini_stream_delta_true():
    client = GeminiClient()
    queue = asyncio.Queue()
    streamer = Streamer("test", queue)
    model = await _pick_gemini_model(client)

    await client.chat_completions(
        model=model,
        messages=[{"role": "user", "content": "Say 'Hello'"}],
        streamer=streamer,
        stream_delta=True,
    )

    partials = []
    while not queue.empty():
        content = StreamContent.model_validate_json(await queue.get())
        if content.type == "partial":
            partials.append(content.value)

    if len(partials) > 1:
        if partials[0]:
            assert not partials[1].startswith(partials[0])
