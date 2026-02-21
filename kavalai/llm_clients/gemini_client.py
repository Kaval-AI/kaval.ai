"""
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import base64
import io
import time
from typing import Any, Dict, List, Optional, Type, Tuple

from google import genai
from google.genai import types
from partial_json_parser import ensure_json
from pydantic import BaseModel
import os
from kavalai.agents.db import ModelCallStat
from kavalai.normalizer import Normalizer, get_default_normalizer
from kavalai.llm_clients.common import (
    create_model_call_stat,
    get_model_name,
    Streamer,
)


class GeminiClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ):
        """
        Initialize the Gemini client.
        """
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")

        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                api_version="v1beta",
            ),
        )

    async def chat_completions(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        response_model: Optional[Type[BaseModel]] = None,
        streamer: Optional[Streamer] = None,
        **kwargs,
    ) -> Tuple[Any, ModelCallStat]:
        """
        Send a chat completion request to Gemini.
        Supports structured output via response_model and streaming via streamer.
        """
        start_time = time.perf_counter()

        model_name = get_model_name(model)
        contents = self._convert_messages(messages)
        config = self._prepare_config(response_model, **kwargs)

        if streamer is None:
            # Non-streaming implementation
            response = await self.client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            duration = time.perf_counter() - start_time
            content = response.text
            if response_model:
                content = response_model.model_validate_json(content)
            stats = self._create_chat_stats(model_name, response, duration)
            return content, stats

        # Streaming implementation
        buffer = io.StringIO()
        last_response = None

        # Call the streaming API
        async for response in await self.client.aio.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=config,
        ):
            last_response = response
            if response.text:
                buffer.write(response.text)
                # Stream partial results if a streamer is provided
                await streamer.stream_partial(ensure_json(buffer.getvalue()))

        # Signal completion of streaming
        await streamer.stream_complete(ensure_json(buffer.getvalue()))

        duration = time.perf_counter() - start_time
        response_text = buffer.getvalue()

        content = response_text
        if response_model:
            # Validate and parse the response text into the requested Pydantic model
            content = response_model.model_validate_json(response_text)

        stats = self._create_chat_stats(model_name, last_response, duration)
        return content, stats

    def _prepare_config(
        self, response_model: Optional[Type[BaseModel]], **kwargs
    ) -> types.GenerateContentConfig:
        """Prepare Gemini generation config, including structured output schema if provided."""
        config_kwargs = {}
        if response_model:
            schema = response_model.model_json_schema()
            self._cleanup_schema(schema)
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = schema

        # Map common kwargs to Gemini specific config fields
        for key in [
            "temperature",
            "top_p",
            "top_k",
            "max_output_tokens",
            "stop_sequences",
        ]:
            if key in kwargs:
                config_kwargs[key] = kwargs.pop(key)

        return types.GenerateContentConfig(**config_kwargs)

    def _create_chat_stats(
        self, model: str, last_response: Any, duration: float
    ) -> ModelCallStat:
        """Create a ModelCallStat object from the Gemini API response metadata."""
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        if last_response and last_response.usage_metadata:
            usage = last_response.usage_metadata
            prompt_tokens = usage.prompt_token_count or 0
            completion_tokens = usage.candidates_token_count or 0
            total_tokens = usage.total_token_count or 0

        return create_model_call_stat(
            call_type="llm",
            model=f"gemini/{model}",
            duration=duration,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            response_data=str(last_response) if last_response else None,
        )

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[types.Content]:
        """Convert OpenAI-style messages to Gemini-style contents."""
        return [self._convert_single_message(m) for m in messages]

    def _convert_single_message(self, message: Dict[str, Any]) -> types.Content:
        """Convert a single OpenAI-style message to a Gemini types.Content object."""
        role = "model" if message["role"] == "assistant" else message["role"]
        parts = []

        content = message.get("content")
        if isinstance(content, list):
            # Handle multi-part content
            for part in content:
                parts.extend(self._convert_content_part(part))
        elif content:
            # Handle simple text content
            parts.append(types.Part(text=content))

        # Handle images if provided as a separate list
        if message.get("images"):
            for img_base64 in message["images"]:
                parts.append(self._create_image_part(img_base64))

        return types.Content(role=role, parts=parts)

    def _convert_content_part(self, part: Dict[str, Any]) -> List[types.Part]:
        """Convert a single part of multi-part content to Gemini types.Part."""
        if part["type"] == "text":
            return [types.Part(text=part["text"])]
        if part["type"] == "image_url":
            url = part["image_url"]["url"]
            if url.startswith("data:image"):
                # Extract base64 data from data URI
                _, base64_data = url.split(",", 1)
                return [self._create_image_part(base64_data)]
        return []

    def _create_image_part(self, base64_data: str) -> types.Part:
        """Create a Gemini image Part from base64 data."""
        return types.Part(
            inline_data=types.Blob(
                mime_type="image/jpeg",
                data=base64.b64decode(base64_data),
            )
        )

    def _cleanup_schema(self, schema: Dict[str, Any]):
        """Recursively remove 'title' and other unsupported fields from JSON schema for Gemini."""
        if not isinstance(schema, dict):
            return

        unsupported = ["title", "description", "default"]
        for field in unsupported:
            if field in schema:
                del schema[field]

        if "properties" in schema:
            for prop in schema["properties"].values():
                self._cleanup_schema(prop)

        if "items" in schema:
            self._cleanup_schema(schema["items"])

    async def generate_image(
        self,
        model: str,
        prompt: str,
        **kwargs,
    ) -> Tuple[str, ModelCallStat]:
        """Generate an image using Gemini models (e.g., gemini-2.5-flash-image)."""
        start_time = time.perf_counter()
        model_name = get_model_name(model)

        try:
            # The example uses generate_content for image generation models
            try:
                response = await self.client.aio.models.generate_content(
                    model=model_name,
                    contents=[prompt],
                    **kwargs,
                )
                duration = time.perf_counter() - start_time

                image_base64 = None
                for part in response.parts:
                    if part.inline_data is not None:
                        image_bytes = part.inline_data.data
                        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                        break
            except Exception:
                image_base64 = None

            if image_base64 is None:
                # Fallback to generate_images if generate_content didn't return an image
                # This might happen for older models or different API versions
                response = await self.client.aio.models.generate_images(
                    model=model_name,
                    prompt=prompt,
                    **kwargs,
                )
                duration = time.perf_counter() - start_time
                image_bytes = response.generated_images[0].image.image_bytes
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            stats = create_model_call_stat(
                call_type="image_generation",
                model=f"gemini/{model_name}",
                duration=duration,
                response_data={"prompt": prompt},
            )
            return image_base64, stats
        except Exception:
            # Re-raise or handle the exception appropriately
            raise

    async def compute_embeddings(
        self,
        model: str,
        texts: List[str],
        normalize: bool = False,
        normalizer: Optional[Normalizer] = None,
        **kwargs,
    ) -> Tuple[List[List[float]], ModelCallStat]:
        """Compute embeddings for a list of texts using Gemini."""
        start_time = time.perf_counter()
        model_name = get_model_name(model)

        response = await self.client.aio.models.embed_content(
            model=model_name,
            contents=texts,
            **kwargs,
        )
        duration = time.perf_counter() - start_time

        embeddings = [emb.values for emb in response.embeddings]
        if normalize:
            if normalizer is None:
                normalizer = get_default_normalizer()
            embeddings = normalizer.transform(embeddings)

        # Estimated tokens if not provided (Gemini embed_content doesn't always return usage in the same way as generate_content)
        total_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            total_tokens = response.usage_metadata.total_token_count or 0
        else:
            # Fallback: estimate tokens (rough estimation: 1 token ~= 4 chars)
            total_tokens = sum(len(t) for t in texts) // 4

        stats = create_model_call_stat(
            call_type="embedding",
            model=f"gemini/{model_name}",
            duration=duration,
            batch_size=len(texts),
            total_tokens=total_tokens,
            response_data=str(response),
        )
        return embeddings, stats

    async def list_models(self) -> List[str]:
        """List available Gemini models, removing the 'models/' prefix."""
        models = []
        async for model in await self.client.aio.models.list():
            # Remove "models/" prefix if present
            name = model.name
            if name.startswith("models/"):
                name = name[len("models/") :]
            models.append(name)
        return models
