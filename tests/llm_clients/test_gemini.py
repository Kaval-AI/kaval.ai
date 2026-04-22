import asyncio
import os
import pytest
from pydantic import BaseModel

from kavalai.llm_clients.gemini_client import GeminiClient, remove_additional_properties
from kavalai.llm_clients.common import Streamer, StreamContent


class SimpleResponse(BaseModel):
    answer: str
    confidence: float


class NestedModel(BaseModel):
    id: int
    name: str


class ComplexResponse(BaseModel):
    items: list[NestedModel]
    count: int
    metadata: dict[str, str]


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


# Unit Tests


def test_remove_additional_properties_simple():
    """Test that additionalProperties is removed from simple schema."""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "additionalProperties": False,
    }
    remove_additional_properties(schema)
    assert "additionalProperties" not in schema
    assert "properties" in schema


def test_remove_additional_properties_nested():
    """Test that additionalProperties is removed from nested schemas."""
    schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "additionalProperties": False,
            }
        },
        "additionalProperties": True,
    }
    remove_additional_properties(schema)
    assert "additionalProperties" not in schema
    assert "additionalProperties" not in schema["properties"]["user"]


def test_remove_additional_properties_with_defs():
    """Test that additionalProperties is removed from $defs."""
    schema = {
        "type": "object",
        "properties": {"item": {"$ref": "#/$defs/Item"}},
        "additionalProperties": False,
        "$defs": {
            "Item": {
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "additionalProperties": False,
            }
        },
    }
    remove_additional_properties(schema)
    assert "additionalProperties" not in schema
    assert "additionalProperties" not in schema["$defs"]["Item"]


def test_remove_additional_properties_pydantic_schema():
    """Test with actual Pydantic model schema."""
    schema = ComplexResponse.model_json_schema()

    # Verify the schema has additionalProperties before removal
    assert "additionalProperties" in str(schema)

    remove_additional_properties(schema)

    # Check that no additionalProperties exists anywhere in the schema
    assert "additionalProperties" not in str(schema)


@pytest.mark.asyncio
async def test_gemini_service_tier_unit():
    """Unit test for service_tier handling in GeminiClient."""
    from unittest.mock import AsyncMock, patch, MagicMock
    from google.genai import types

    messages = [{"role": "user", "content": "test"}]

    # Mock the SDK client
    with patch("google.genai.Client") as mock_genai_client:
        mock_instance = mock_genai_client.return_value
        mock_instance.aio.models.generate_content_stream = AsyncMock()
        # Mocking a chunk with usage metadata so the loop finishes and returns stats
        mock_chunk = MagicMock()
        mock_chunk.candidates = []
        mock_chunk.usage_metadata.prompt_token_count = 10
        mock_chunk.usage_metadata.candidates_token_count = 5

        async def mock_gen():
            yield mock_chunk

        mock_instance.aio.models.generate_content_stream.return_value = mock_gen()

        client = GeminiClient(api_key="fake")

        # 1. Test priority mapping via LLMClient
        from kavalai.llm_clients.llm_client import LLMClient

        llm_client = LLMClient("gemini/gemini-2.0-flash")
        llm_client.client = client

        # This should work without Pydantic validation error
        # GeminiClient maps priority to service_tier enum
        await llm_client.chat_completions(messages=messages, priority="high")

        # Verify that service_tier was passed as ServiceTier enum
        args, kwargs = mock_instance.aio.models.generate_content_stream.call_args
        config = kwargs.get("config")
        assert isinstance(config, types.GenerateContentConfig)
        # Check that service_tier is set to the correct enum value
        assert config.service_tier == types.ServiceTier.PRIORITY
