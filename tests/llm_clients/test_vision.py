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
    # Mock the async context manager
    mock_stream.__aenter__.return_value = [
        MagicMock(
            delta='{"answer": "A screenshot of a webpage"}',
            __class__=type(
                "ResponseTextDeltaEvent",
                (),
                {"delta": '{"answer": "A screenshot of a webpage"}'},
            ),
        ),
        MagicMock(
            response=MagicMock(usage=MagicMock(input_tokens=10, output_tokens=5)),
            __class__=type(
                "ResponseCompletedEvent",
                (),
                {
                    "response": MagicMock(
                        usage=MagicMock(input_tokens=10, output_tokens=5)
                    )
                },
            ),
        ),
    ]

    # Actually OpenAI client uses a custom stream object, let's simplify mocking
    # by patching the stream method to return an async iterator

    async def mock_stream_iter():
        yield MagicMock(delta='{"answer": "A screenshot', __class__=MagicMock)
        yield MagicMock(delta=' of a webpage"}', __class__=MagicMock)
        yield MagicMock(
            response=MagicMock(usage=MagicMock(input_tokens=10, output_tokens=5)),
            __class__=MagicMock,
        )

    with patch.object(client.client.responses, "stream") as mock_stream_method:
        mock_stream_method.return_value.__aenter__.return_value = mock_stream_iter()

        # Patching isinstance check in openai_client.py
        with patch(
            "kavalai.llm_clients.openai_client.ResponseTextDeltaEvent", new=MagicMock
        ), patch(
            "kavalai.llm_clients.openai_client.ResponseCompletedEvent", new=MagicMock
        ):
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
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 5
    mock_response.usage_metadata.total_token_count = 15

    async def mock_stream(*args, **kwargs):
        yield mock_response

    with patch.object(
        client.client.aio.models, "generate_content_stream"
    ) as mock_generate:
        mock_generate.return_value = mock_stream()

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
        contents = call_kwargs["contents"]
        assert len(contents) == 1
        assert contents[0].role == "user"
        assert len(contents[0].parts) == 2
        assert contents[0].parts[0].text == "What is in this image?"
        assert contents[0].parts[1].inline_data.data == b"Hello"
        assert contents[0].parts[1].inline_data.mime_type == "image/jpeg"
