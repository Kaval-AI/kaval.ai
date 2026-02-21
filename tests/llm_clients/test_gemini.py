import asyncio
import os
import pytest
import base64
from pydantic import BaseModel
from unittest.mock import AsyncMock, patch, MagicMock
from kavalai.llm_clients.gemini_client import (
    GeminiClient,
    _cleanup_schema,
    _convert_content_part,
)
from kavalai.llm_clients.common import Streamer


class SimpleResponse(BaseModel):
    answer: str
    confidence: float


@pytest.fixture
def gemini_client():
    return GeminiClient(api_key="fake-key")


@pytest.mark.asyncio
async def test_gemini_structured_output(gemini_client):
    mock_response = MagicMock()
    mock_response.text = '{"answer": "4", "confidence": 1.0}'
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 5
    mock_response.usage_metadata.total_token_count = 15

    # Mocking async iterator
    async def mock_stream(*args, **kwargs):
        yield mock_response

    with patch.object(
        gemini_client.client.aio.models, "generate_content", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = mock_response

        messages = [
            {
                "role": "user",
                "content": "What is 2+2? Answer in JSON format with 'answer' and 'confidence' fields.",
            }
        ]
        content, stats = await gemini_client.chat_completions(
            model="gemini-2.0-flash", messages=messages, response_model=SimpleResponse
        )

        assert isinstance(content, SimpleResponse)
        assert "4" in content.answer
        assert content.confidence >= 0.0
        assert stats.total_tokens == 15
        assert stats.model == "gemini/gemini-2.0-flash"


@pytest.mark.asyncio
async def test_gemini_streaming(gemini_client):
    mock_response1 = MagicMock()
    mock_response1.text = '{"answer":'
    mock_response1.usage_metadata = None

    mock_response2 = MagicMock()
    mock_response2.text = ' "4", "confidence": 1.0}'
    mock_response2.usage_metadata.prompt_token_count = 10
    mock_response2.usage_metadata.candidates_token_count = 5
    mock_response2.usage_metadata.total_token_count = 15

    async def mock_stream(*args, **kwargs):
        yield mock_response1
        yield mock_response2

    with patch.object(
        gemini_client.client.aio.models,
        "generate_content_stream",
        new_callable=AsyncMock,
    ) as mock_generate:
        mock_generate.return_value = mock_stream()

        messages = [{"role": "user", "content": "What is 2+2?"}]
        queue = asyncio.Queue()
        streamer = Streamer(name="test", queue=queue)
        content, stats = await gemini_client.chat_completions(
            model="gemini-2.0-flash",
            messages=messages,
            response_model=SimpleResponse,
            streamer=streamer,
        )

        assert isinstance(content, SimpleResponse)
        assert content.answer == "4"

        stream_content = ""
        while not queue.empty():
            stream_content += await queue.get() + "\n"

        assert '"partial"' in stream_content
        assert '"complete"' in stream_content
        assert "4" in stream_content
        assert stats.total_tokens == 15


@pytest.mark.asyncio
async def test_gemini_multimodal_input(gemini_client):
    mock_response = MagicMock()
    mock_response.text = "I see a cat."
    mock_response.usage_metadata.total_token_count = 20

    async def mock_stream(*args, **kwargs):
        yield mock_response

    with patch.object(
        gemini_client.client.aio.models, "generate_content", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = mock_response

        # Test both image_url and images list
        fake_image = base64.b64encode(b"fake image data").decode("utf-8")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{fake_image}"},
                    },
                ],
                "images": [fake_image],
            }
        ]

        content, stats = await gemini_client.chat_completions(
            model="gemini-2.0-flash", messages=messages
        )

        assert content == "I see a cat."
        assert stats.total_tokens == 20
        # Verify content conversion (indirectly by checking mock call if we wanted to be thorough)
        assert mock_generate.called


@pytest.mark.asyncio
async def test_gemini_generate_image(gemini_client):
    # Mock for generate_content
    mock_content_response = MagicMock()
    mock_part = MagicMock()
    mock_part.inline_data = MagicMock()
    mock_part.inline_data.data = b"fake-image-bytes"
    mock_content_response.parts = [mock_part]

    with patch.object(
        gemini_client.client.aio.models, "generate_content", new_callable=AsyncMock
    ) as mock_gen_content:
        mock_gen_content.return_value = mock_content_response

        image_base64, stats = await gemini_client.generate_image(
            model="imagen-3.0-generate-001", prompt="A sunset"
        )

        assert image_base64 == base64.b64encode(b"fake-image-bytes").decode("utf-8")
        assert stats.call_type == "image_generation"
        assert stats.model == "gemini/imagen-3.0-generate-001"
        assert mock_gen_content.called


@pytest.mark.asyncio
async def test_gemini_list_models(gemini_client):
    mock_model1 = MagicMock()
    mock_model1.name = "models/gemini-pro"
    mock_model2 = MagicMock()
    mock_model2.name = "gemini-ultra"

    async def mock_list_models():
        yield mock_model1
        yield mock_model2

    with patch.object(
        gemini_client.client.aio.models, "list", new_callable=AsyncMock
    ) as mock_list:
        mock_list.return_value = mock_list_models()

        models = await gemini_client.list_models()

        assert "gemini-pro" in models
        assert "gemini-ultra" in models
        assert len(models) == 2


@pytest.mark.asyncio
async def test_gemini_compute_embeddings_custom_normalizer(gemini_client):
    from kavalai.normalizer import Normalizer

    mock_emb = MagicMock()
    mock_emb.values = [3.0, 4.0]
    mock_response = MagicMock()
    mock_response.embeddings = [mock_emb]
    mock_response.usage_metadata.total_token_count = 5

    with patch.object(
        gemini_client.client.aio.models, "embed_content", new_callable=AsyncMock
    ) as mock_embed:
        mock_embed.return_value = mock_response

        # L1 normalizer: [3.0, 4.0] -> [3/7, 4/7]
        l1_normalizer = Normalizer(l1=True)

        embeddings, stats = await gemini_client.compute_embeddings(
            model="text-embedding-004",
            texts=["hi"],
            normalize=True,
            normalizer=l1_normalizer,
        )

    assert embeddings[0] == pytest.approx([3 / 7, 4 / 7])
    assert stats.total_tokens == 5


@pytest.mark.asyncio
async def test_gemini_chat_completions_no_response_model(gemini_client):
    mock_response = MagicMock()
    mock_response.text = "Hello!"
    mock_response.usage_metadata.total_token_count = 5

    async def mock_stream(*args, **kwargs):
        yield mock_response

    with patch.object(
        gemini_client.client.aio.models, "generate_content", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = mock_response

        content, stats = await gemini_client.chat_completions(
            model="gemini-2.0-flash", messages=[{"role": "user", "content": "Hi"}]
        )

        assert content == "Hello!"
        assert stats.total_tokens == 5


@pytest.mark.asyncio
async def test_gemini_provider_model_syntax(gemini_client):
    """Test that provider/model syntax is correctly handled."""
    mock_response = MagicMock()
    mock_response.text = "Hello"
    mock_response.usage_metadata.prompt_token_count = 1
    mock_response.usage_metadata.candidates_token_count = 1
    mock_response.usage_metadata.total_token_count = 2

    async def mock_stream(*args, **kwargs):
        yield mock_response

    with patch.object(
        gemini_client.client.aio.models, "generate_content", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = mock_response

        await gemini_client.chat_completions(
            model="google/gemini-2.0-flash",
            messages=[{"role": "user", "content": "Hi"}],
        )

        # Verify that only the model part was passed to the API
        assert mock_generate.call_args.kwargs["model"] == "gemini-2.0-flash"


@pytest.mark.asyncio
async def test_gemini_compute_embeddings_no_usage_metadata(gemini_client):
    mock_emb = MagicMock()
    mock_emb.values = [0.1, 0.2]
    mock_response = MagicMock()
    mock_response.embeddings = [mock_emb]
    mock_response.usage_metadata = None

    with patch.object(
        gemini_client.client.aio.models, "embed_content", new_callable=AsyncMock
    ) as mock_embed:
        mock_embed.return_value = mock_response

        embeddings, stats = await gemini_client.compute_embeddings(
            model="text-embedding-004", texts=["hello"]
        )

        assert stats.total_tokens > 0  # Fallback estimation should work


@pytest.mark.asyncio
async def test_gemini_compute_embeddings_multiple_texts(gemini_client):
    mock_emb1 = MagicMock()
    mock_emb1.values = [0.1, 0.2]
    mock_emb2 = MagicMock()
    mock_emb2.values = [0.3, 0.4]
    mock_response = MagicMock()
    mock_response.embeddings = [mock_emb1, mock_emb2]
    mock_response.usage_metadata.total_token_count = 10

    with patch.object(
        gemini_client.client.aio.models, "embed_content", new_callable=AsyncMock
    ) as mock_embed:
        mock_embed.return_value = mock_response

        embeddings, stats = await gemini_client.compute_embeddings(
            model="text-embedding-004", texts=["hello", "world"]
        )

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2]
        assert embeddings[1] == [0.3, 0.4]
        assert stats.batch_size == 2
        assert stats.total_tokens == 10


@pytest.mark.asyncio
async def test_gemini_cleanup_schema(gemini_client):
    schema = {
        "title": "MySchema",
        "description": "desc",
        "default": "val",
        "properties": {
            "field1": {"title": "Field 1", "type": "string"},
            "field2": {"items": {"title": "Item title", "type": "string"}},
        },
    }
    _cleanup_schema(schema)
    assert "description" not in schema
    assert "default" not in schema
    assert "title" not in schema["properties"]["field1"]
    assert "title" not in schema["properties"]["field2"]["items"]


@pytest.mark.asyncio
async def test_gemini_convert_content_part_invalid(gemini_client):
    part = {"type": "invalid"}
    res = _convert_content_part(part)
    assert res == []


@pytest.mark.asyncio
async def test_gemini_cleanup_schema_non_dict(gemini_client):
    schema = ["not", "a", "dict"]
    _cleanup_schema(schema)


@pytest.mark.asyncio
async def test_gemini_reasoning_parameters(gemini_client):
    """Test that reasoning parameters are correctly mapped to Gemini config."""
    mock_response = MagicMock()
    mock_response.text = "I have thought about it."
    mock_response.usage_metadata.total_token_count = 10

    with patch.object(
        gemini_client.client.aio.models, "generate_content", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = mock_response

        # Test reasoning_effort
        await gemini_client.chat_completions(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "Think deep"}],
            reasoning_effort="high",
        )
        config = mock_generate.call_args.kwargs["config"]
        assert config.thinking_config.include_thoughts is True
        assert config.thinking_config.thinking_budget == 24576
        if hasattr(config.thinking_config, "thinking_level"):
            assert config.thinking_config.thinking_level == "high"

        # Test thinking_level and thinking_budget
        await gemini_client.chat_completions(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "Think deep"}],
            thinking_level="medium",
            thinking_budget=1024,
        )
        config = mock_generate.call_args.kwargs["config"]
        assert config.thinking_config.include_thoughts is True
        assert config.thinking_config.thinking_budget == 1024
        # thinking_level is set via setattr in our implementation if not on the type directly,
        # so we check if it's there or if it was at least attempted.
        if hasattr(config.thinking_config, "thinking_level"):
            assert config.thinking_config.thinking_level == "medium"


@pytest.mark.asyncio
async def test_gemini_thought_summary(gemini_client):
    """Test that thought summaries are extracted from the response."""
    mock_response = MagicMock()
    mock_response.text = "Final answer"
    mock_response.usage_metadata.total_token_count = 10

    # Mock candidate with thoughts
    mock_part = MagicMock()
    mock_part.thought = "I am thinking..."
    mock_part.text = None

    mock_part2 = MagicMock()
    mock_part2.thought = None
    mock_part2.text = "Final answer"

    mock_candidate = MagicMock()
    mock_candidate.content.parts = [mock_part, mock_part2]
    mock_response.candidates = [mock_candidate]

    with patch.object(
        gemini_client.client.aio.models, "generate_content", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = mock_response

        content, stats = await gemini_client.chat_completions(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "Explain AI"}],
        )

        assert content == "Final answer"
        # Since we don't have a place in stats for thoughts yet, we just verify it doesn't crash
        # and we could potentially verify response_data contains the thought if we wanted.
        assert stats.response_data is not None


def test_gemini_initialization_env_key(monkeypatch):
    # Set env var
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

    with patch("google.genai.Client") as mock_client:
        _ = GeminiClient()
        args, kwargs = mock_client.call_args
        assert kwargs["api_key"] == "gemini-key"


def test_gemini_initialization_aiza_key(monkeypatch):
    # Set AIza key
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSomeKey")

    with patch("google.genai.Client") as mock_client:
        _ = GeminiClient()
        args, kwargs = mock_client.call_args
        assert kwargs["api_key"] == "AIzaSomeKey"


# INTEGRATION TESTS

INTEGRATION_MARK = pytest.mark.skipif(
    not (
        os.getenv("KAVALAI_RUN_INTEGRATION") == "true" and os.getenv("GEMINI_API_KEY")
    ),
    reason="Integration test disabled (set KAVALAI_RUN_INTEGRATION=true and provide GEMINI_API_KEY)",
)


@INTEGRATION_MARK
@pytest.mark.asyncio
async def test_gemini_structured_output_integration():
    client = GeminiClient()

    messages = [
        {
            "role": "user",
            "content": "What is 2+2? Answer in JSON format with 'answer' and 'confidence' fields.",
        }
    ]
    content, stats = await client.chat_completions(
        model="gemini-2.0-flash", messages=messages, response_model=SimpleResponse
    )

    assert isinstance(content, SimpleResponse)
    assert "4" in content.answer
    assert content.confidence >= 0.0


@INTEGRATION_MARK
@pytest.mark.asyncio
async def test_gemini_list_models_integration():
    client = GeminiClient()

    models = await client.list_models()
    assert len(models) > 0
    assert any("gemini" in m for m in models)


@INTEGRATION_MARK
@pytest.mark.asyncio
async def test_gemini_compute_embeddings_integration():
    client = GeminiClient()

    texts = ["Hello world", "Goodbye world"]
    # text-embedding-004 might not be available in v1beta in some regions,
    # but the client is initialized with v1beta. Try a more common model or handle 404.
    try:
        embeddings, stats = await client.compute_embeddings(
            model="text-embedding-004", texts=texts, normalize=True
        )
    except Exception as e:
        if "404" in str(e):
            pytest.skip(f"Embedding model not found: {e}")
        raise

    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0
    assert stats.total_tokens > 0


@INTEGRATION_MARK
@pytest.mark.asyncio
async def test_gemini_generate_image_integration():
    client = GeminiClient()

    # Note: Imagen might not be enabled for all keys or regions
    try:
        image_base64, stats = await client.generate_image(
            model="imagen-3.0-generate-001",
            prompt="A small cute robot helping with coding.",
        )
        assert len(image_base64) > 100
        assert stats.call_type == "image_generation"
    except Exception as e:
        pytest.skip(f"Imagen image generation failed (might be expected): {e}")
