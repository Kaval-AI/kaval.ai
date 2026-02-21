import asyncio

import pytest
from pydantic import BaseModel

from kavalai.llm_clients.common import Streamer, StreamContent
from kavalai.llm_clients.gemini_client import GeminiClient


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


@pytest.mark.asyncio
async def test_gemini_chat_completions_simple(gemini_client):
    """Test simple message and string result."""
    messages = [{"role": "user", "content": "Say 'Hello'"}]
    model = await _pick_chat_model(gemini_client)
    result, stats = await gemini_client.chat_completions(model=model, messages=messages)

    assert "Hello" in result
    assert stats.prompt_tokens > 0
    assert stats.completion_tokens > 0
    assert stats.model.startswith("gemini/")
    assert stats.cost >= 0
    assert stats.currency == "USD"


@pytest.mark.asyncio
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
async def test_gemini_chat_completions_streamer(gemini_client):
    """Test streamer functionality."""
    messages = [{"role": "user", "content": "Say 'Hi'"}]

    # Create async queue and real streamer
    queue = asyncio.Queue()
    streamer = Streamer("test_streamer", queue)

    # Make API call with streamer
    model = await _pick_chat_model(gemini_client)
    result, stats = await gemini_client.chat_completions(
        model=model, messages=messages, streamer=streamer
    )

    # Collect all streamed content
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
async def test_gemini_generate_image(gemini_client):
    """Test image generation."""
    prompt = "A simple red square 1x1 size."
    image_base64, stats = await gemini_client.generate_image(
        model="gemini-2.5-flash-image",  # Or another model that might support it if any
        prompt=prompt,
    )
    if image_base64:
        assert isinstance(image_base64, str)
        assert stats.call_type == "image_generation"
        assert stats.cost >= 0
        assert stats.currency == "USD"


@pytest.mark.asyncio
async def test_gemini_compute_embeddings(gemini_client):
    """Test embedding generation."""
    texts = ["Hello world", "How are you?"]

    # Pick an available embedding model dynamically
    models = await gemini_client.list_models()
    emb_models = [m for m in models if "embedding" in m.lower()]
    if not emb_models:
        pytest.skip("No embedding model available in this environment")
    emb_model = emb_models[0]

    embeddings, stats = await gemini_client.compute_embeddings(
        model=emb_model, texts=texts
    )
    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0
    assert stats.call_type == "embedding"
    assert stats.cost >= 0
    assert stats.currency == "USD"


@pytest.mark.asyncio
async def test_gemini_chat_completions_streamer_with_response_model(gemini_client):
    """Test streamer functionality together with structured output (response_model)."""
    messages = [
        {
            "role": "user",
            "content": "What is the capital of France? Return JSON with fields: answer (string) and confidence (number).",
        }
    ]

    # Create async queue and real streamer
    queue = asyncio.Queue()
    streamer = Streamer("test_streamer_json", queue)

    # Make API call with streamer and response_model
    model = await _pick_chat_model(gemini_client)
    result, stats = await gemini_client.chat_completions(
        model=model, messages=messages, response_model=SimpleResponse, streamer=streamer
    )

    # Collect all streamed content
    partials = []
    final_streamed = None

    while not queue.empty():
        content_json = await queue.get()
        content = StreamContent.model_validate_json(content_json)
        if content.type == "partial":
            partials.append(content.value)
        elif content.type == "complete":
            final_streamed = content.value

    # Result must be parsed into the Pydantic model
    assert isinstance(result, SimpleResponse)
    assert "Paris" in result.answer
    assert result.confidence >= 0

    # We should have received partial JSON chunks
    assert len(partials) > 0
    # The final streamed value should be valid JSON for the model and match the final result
    assert final_streamed is not None
    final_parsed = SimpleResponse.model_validate_json(final_streamed)
    assert final_parsed.answer == result.answer
    assert pytest.approx(final_parsed.confidence, rel=1e-3) == result.confidence
