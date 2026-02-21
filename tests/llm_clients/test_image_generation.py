import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from kavalai.llm_clients.openai_client import OpenAIClient
from kavalai.llm_clients.gemini_client import GeminiClient


@pytest.mark.asyncio
async def test_openai_generate_image():
    client = OpenAIClient(api_key="fake-key")

    mock_response = MagicMock()
    mock_image = MagicMock()
    mock_image.b64_json = "SGVsbG8="  # "Hello" in base64
    mock_response.data = [mock_image]

    with patch.object(
        client.client.images, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = mock_response

        img_b64, stats = await client.generate_image(
            model="dalle-3", prompt="A futuristic city"
        )

        assert img_b64 == "SGVsbG8="
        assert stats.call_type == "image_generation"
        assert stats.model == "openai/dalle-3"
        mock_generate.assert_called_once()


@pytest.mark.asyncio
async def test_gemini_generate_image():
    client = GeminiClient(api_key="fake-key")

    mock_response = MagicMock()
    mock_part = MagicMock()
    mock_part.inline_data.data = b"Hello"
    mock_response.parts = [mock_part]

    with patch.object(
        client.client.aio.models, "generate_content", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = mock_response

        img_b64, stats = await client.generate_image(
            model="imagen-3.0-generate-002", prompt="A serene landscape"
        )

        assert img_b64 == "SGVsbG8="  # base64 of b"Hello"
        assert stats.call_type == "image_generation"
        assert stats.model == "gemini/imagen-3.0-generate-002"
        mock_generate.assert_called_once()
        assert mock_generate.call_args[1]["http_options"] == {"timeout": 30.0}
