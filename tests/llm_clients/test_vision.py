import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pydantic import BaseModel
from kavalai.llm_clients.openai_client import OpenAIClient
from kavalai.llm_clients.gemini_client import GeminiClient


class SimpleResponse(BaseModel):
    answer: str


@pytest.mark.asyncio
async def test_openai_chat_completions_with_images():
    client = OpenAIClient(api_key="fake-key")

    mock_stream = AsyncMock()

    async def mock_stream_iter():
        chunk1 = MagicMock()
        chunk1.type = "content.delta"
        chunk1.delta = '{"answer": "A screenshot'
        yield chunk1
        chunk2 = MagicMock()
        chunk2.type = "content.delta"
        chunk2.delta = ' of a webpage"}'
        yield chunk2

    final_completion = MagicMock()
    final_completion.usage = MagicMock()
    final_completion.usage.prompt_tokens = 10
    final_completion.usage.completion_tokens = 5
    mock_stream.get_final_completion = AsyncMock(return_value=final_completion)

    with patch.object(
        client.client.beta.chat.completions, "stream"
    ) as mock_stream_method:
        # Create a class that mimics the stream object
        class MockStream:
            def __init__(self, iter_func):
                self.iter_func = iter_func
                self.get_final_completion = AsyncMock(return_value=final_completion)

            def __aiter__(self):
                return self.iter_func()

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        mock_stream_method.return_value = MockStream(mock_stream_iter)

        messages = [
            {
                "role": "user",
                "content": "What is in this image?",
                "images": ["SGVsbG8="],  # "Hello" in base64
            }
        ]

        content, stats = await client.chat_completions(
            model="gpt-4o", messages=messages, response_model=SimpleResponse
        )

        assert content.answer == "A screenshot of a webpage"
        # Verify that messages were formatted correctly for OpenAI
        call_args = mock_stream_method.call_args[1]
        formatted_messages = call_args["messages"]
        assert len(formatted_messages) == 1
        assert formatted_messages[0]["role"] == "user"
        assert isinstance(formatted_messages[0]["content"], list)
        assert formatted_messages[0]["content"][0]["type"] == "text"
        assert formatted_messages[0]["content"][1]["type"] == "image_url"
        assert (
            "data:image/jpeg;base64,SGVsbG8="
            in formatted_messages[0]["content"][1]["image_url"]["url"]
        )


@pytest.mark.asyncio
async def test_gemini_chat_completions_with_images():
    client = GeminiClient(api_key="fake-key")

    mock_response = AsyncMock()
    mock_response.text = '{"answer": "A screenshot"}'
    mock_part = MagicMock()
    mock_part.text = '{"answer": "A screenshot"}'
    mock_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 5
    mock_response.usage_metadata.total_token_count = 15

    async def mock_stream(*args, **kwargs):
        yield mock_response

    with patch.object(
        client.client.aio.models, "generate_content", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = mock_response

        messages = [
            {
                "role": "user",
                "content": "What is in this image?",
                "images": ["SGVsbG8="],
            }
        ]

        content, stats = await client.chat_completions(
            model="gemini-2.0-flash", messages=messages, response_model=SimpleResponse
        )

        assert content.answer == "A screenshot"
        # Verify Gemini formatting
        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["http_options"] == {"timeout": 30.0}
        contents = call_kwargs["contents"]
        assert len(contents) == 1
        assert contents[0].role == "user"
        assert len(contents[0].parts) == 2
        assert contents[0].parts[0].text == "What is in this image?"
        assert contents[0].parts[1].inline_data.data == b"Hello"
        assert contents[0].parts[1].inline_data.mime_type == "image/jpeg"
