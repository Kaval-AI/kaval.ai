import asyncio
import os
import pytest
from pydantic import BaseModel

from kavalai.llm_clients.gemini_client import GeminiClient
from kavalai.llm_clients.common import Streamer, StreamContent


class SimpleResponse(BaseModel):
    answer: str
    confidence: float


@pytest.fixture
def gemini_client():
    return GeminiClient()


async def _pick_chat_model(client: GeminiClient) -> str:
    models = await client.list_models()
    for cand in models:
        name = cand.lower()
        if "flash" in name and "image" not in name and "generate" not in name:
            return cand
    for cand in models:
        name = cand.lower()
        if "image" not in name and "generate" not in name:
            return cand
    return models[0] if models else "gemini-1.5-flash"


# Integration Tests


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
async def test_gemini_chat_completions_simple(gemini_client):
    """Test simple message and string result."""
    messages = [{"role": "user", "content": "Say 'Hello'"}]
    model = await _pick_chat_model(gemini_client)
    result, stats = await gemini_client.chat_completions(model=model, messages=messages)

    assert "Hello" in result
    assert stats.prompt_tokens > 0
    assert stats.completion_tokens > 0
    assert stats.model.startswith("gemini/")
    assert stats.cost is None
    assert stats.currency is None


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
async def test_gemini_chat_completions_formatted(gemini_client):
    """Test formatted result (structured output)."""
    messages = [
        {
            "role": "user",
            "content": "What is the capital of France? Return JSON with answer and confidence.",
        }
    ]
    model = await _pick_chat_model(gemini_client)
    result, stats = await gemini_client.chat_completions(
        model=model, messages=messages, response_model=SimpleResponse
    )

    assert isinstance(result, SimpleResponse)
    assert "Paris" in result.answer
    assert result.confidence > 0
    assert stats.prompt_tokens > 0
    assert stats.completion_tokens > 0


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
async def test_gemini_chat_completions_multimodal(gemini_client):
    """Test results with text and image input."""
    # Small 1x1 black PNG
    dummy_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{dummy_image_b64}"},
                },
            ],
        }
    ]
    model = await _pick_chat_model(gemini_client)
    result, stats = await gemini_client.chat_completions(model=model, messages=messages)

    assert isinstance(result, str)
    assert len(result) > 0
    assert stats.prompt_tokens > 0


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
async def test_gemini_chat_completions_streamer(gemini_client):
    """Test streamer functionality."""
    messages = [{"role": "user", "content": "Say 'Hi'"}]
    queue = asyncio.Queue()
    streamer = Streamer("test_streamer", queue)

    model = await _pick_chat_model(gemini_client)
    result, stats = await gemini_client.chat_completions(
        model=model, messages=messages, streamer=streamer
    )

    partials = []
    final_streamed = None
    while not queue.empty():
        content_json = await queue.get()
        content = StreamContent.model_validate_json(content_json)
        if content.type == "partial":
            partials.append(content.value)
        elif content.type == "complete":
            final_streamed = content.value

    assert "Hi" in result
    assert len(partials) > 0
    assert final_streamed == result


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
async def test_gemini_stream_delta(gemini_client):
    """Test stream_delta functionality."""
    messages = [{"role": "user", "content": "Say 'Hello'"}]
    model = await _pick_chat_model(gemini_client)

    # Test delta=False (default)
    queue = asyncio.Queue()
    streamer = Streamer("test_no_delta", queue)
    await gemini_client.chat_completions(
        model=model, messages=messages, streamer=streamer, stream_delta=False
    )
    partials_no_delta = []
    while not queue.empty():
        c = StreamContent.model_validate_json(await queue.get())
        if c.type == "partial":
            partials_no_delta.append(c.value)

    if len(partials_no_delta) > 1:
        assert partials_no_delta[1].startswith(partials_no_delta[0])

    # Test delta=True
    queue = asyncio.Queue()
    streamer = Streamer("test_delta", queue)
    await gemini_client.chat_completions(
        model=model, messages=messages, streamer=streamer, stream_delta=True
    )
    partials_delta = []
    while not queue.empty():
        c = StreamContent.model_validate_json(await queue.get())
        if c.type == "partial":
            partials_delta.append(c.value)

    if len(partials_delta) > 1 and partials_delta[0]:
        assert not partials_delta[1].startswith(partials_delta[0])


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
async def test_gemini_compute_embeddings(gemini_client):
    """Test embedding generation."""
    texts = ["Hello world", "How are you?"]
    models = await gemini_client.list_models()
    emb_models = [m for m in models if "embedding" in m.lower()]
    if not emb_models:
        pytest.skip("No embedding model available")
    emb_model = emb_models[0]

    embeddings, stats = await gemini_client.compute_embeddings(
        model=emb_model, texts=texts
    )
    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0
    assert stats.call_type == "embedding"
    assert stats.model.startswith("gemini/")


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
async def test_gemini_thinking_streaming(gemini_client):
    """Test real-time thought streaming for Gemini."""
    # Find a model that supports thinking (e.g. 2.0-flash-thinking)
    models = await gemini_client.list_models()
    thinking_model = next((m for m in models if "thinking" in m.lower()), None)
    if not thinking_model:
        pytest.skip("No thinking model available")

    messages = [{"role": "user", "content": "Solve 12345 * 67890 step by step."}]
    queue = asyncio.Queue()
    streamer = Streamer("test_thinking", queue)

    await gemini_client.chat_completions(
        model=thinking_model, messages=messages, streamer=streamer, thinking_budget=1024
    )

    thoughts = []
    content = []
    while not queue.empty():
        c = StreamContent.model_validate_json(await queue.get())
        if c.name == "thought" and c.type == "partial":
            thoughts.append(c.value)
        elif c.name == "test_thinking" and c.type == "partial":
            content.append(c.value)

    assert len(thoughts) > 0
    assert len(content) > 0
