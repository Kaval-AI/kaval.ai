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


def _prepare_config(
    response_model: Optional[Type[BaseModel]], **kwargs
) -> types.GenerateContentConfig:
    """Prepare Gemini generation config."""
    config_kwargs = {}
    if response_model:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_model

    # Handle Reasoning Parameters
    effort = kwargs.pop("reasoning_effort", None)
    level = kwargs.pop("thinking_level", None)
    budget = kwargs.pop("thinking_budget", None)

    if effort or level or budget:
        include = effort != "none"
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            include_thoughts=include
        )
        if include:
            if effort and not (level or budget):
                budget_map = {
                    "minimal": 1024,
                    "low": 1024,
                    "medium": 8192,
                    "high": 24576,
                }
                config_kwargs["thinking_config"].thinking_budget = budget_map.get(
                    effort
                )
                config_kwargs["thinking_config"].thinking_level = (
                    effort if effort in ["minimal", "low", "medium", "high"] else None
                )
            if budget:
                config_kwargs["thinking_config"].thinking_budget = budget
            if level:
                config_kwargs["thinking_config"].thinking_level = level

    # Map common kwargs
    for key in ["temperature", "top_p", "top_k", "max_output_tokens", "stop_sequences"]:
        if key in kwargs:
            config_kwargs[key] = kwargs.pop(key)

    return types.GenerateContentConfig(**config_kwargs)


def _create_chat_stats(
    model: str, last_response: Any, duration: float
) -> ModelCallStat:
    """Create a ModelCallStat object from the Gemini API response metadata."""
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    thought_summary = None

    if last_response and last_response.usage_metadata:
        usage = last_response.usage_metadata
        prompt_tokens = usage.prompt_token_count or 0
        completion_tokens = usage.candidates_token_count or 0
        total_tokens = usage.total_token_count or 0

    # Extract thought summaries if available
    # Thoughts are typically in response.candidates[0].content.parts
    if last_response and last_response.candidates:
        candidate = last_response.candidates[0]
        thought_summary = (
            "\n".join(
                part.thought
                for part in (candidate.content.parts or [])
                if getattr(part, "thought", None)
            )
            or None
        )

    stat = create_model_call_stat(
        call_type="llm",
        model=f"gemini/{model}",
        duration_sections=duration,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        response_data=str(last_response) if last_response else None,
    )

    if thought_summary:
        # We can store thought summary in the stat if needed,
        # for now let's just make sure it's accessible or logged if necessary.
        # ModelCallStat might not have a dedicated thought field,
        # but it can be part of response_data.
        pass

    return stat


def _convert_content_part(part: Dict[str, Any]) -> Optional[types.Part]:
    """Convert a single part of multi-part content to Gemini types.Part."""
    if part["type"] == "text":
        return types.Part(text=part["text"])
    if part["type"] == "image_url":
        url = part["image_url"]["url"]
        if url.startswith("data:image"):
            # Extract base64 data from data URI
            _, b64 = url.split(",", 1)
            return types.Part(
                inline_data=types.Blob(
                    mime_type="image/jpeg",
                    data=base64.b64decode(b64),
                )
            )
    return None


def _convert_single_message(message: Dict[str, Any]) -> types.Content:
    """
    Convert a single OpenAI-style message to a Gemini types.Content object.
    """
    role = "model" if message["role"] == "assistant" else message["role"]
    parts = []

    content = message.get("content")
    if isinstance(content, list):
        # Handle multi-part content
        for p in content:
            converted = _convert_content_part(p)
            if converted:
                parts.append(converted)
    elif content:
        # Handle simple text content
        parts.append(types.Part(text=content))

    # Handle images if provided as a separate list
    for img_base64 in message.get("images", []):
        parts.append(
            types.Part(
                inline_data=types.Blob(
                    mime_type="image/jpeg",
                    data=base64.b64decode(img_base64),
                )
            )
        )

    return types.Content(role=role, parts=parts)


def _convert_messages(messages: List[Dict[str, Any]]) -> List[types.Content]:
    """Convert OpenAI-style messages to Gemini-style contents."""
    return [_convert_single_message(m) for m in messages]


class GeminiClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the Gemini client.

        Args:
            api_key (Optional[str]): The API key for Gemini. If not provided,
                will attempt to get from GEMINI_API_KEY environment variable.
            timeout (float): Default timeout for API calls in seconds.

        Raises:
            ValueError: If no API key is provided or found in environment
                variables.
        """
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "Gemini API key must be provided either through the "
                "api_key parameter or GEMINI_API_KEY environment variable"
            )

        self.timeout = timeout
        self.client = genai.Client(
            api_key=api_key,
        )

    async def chat_completions(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        response_model: Optional[Type[BaseModel]] = None,
        streamer: Optional[Streamer] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Tuple[Any, ModelCallStat]:
        """
        Send a chat completion request to Gemini.
        Supports structured output via response_model and streaming via
        streamer.
        """
        start_time = time.perf_counter()
        effective_timeout = timeout if timeout is not None else self.timeout
        model_name = get_model_name(model)

        # Remove system message and put it into config if it exists
        system_instruction = None
        other_messages = []
        for m in messages:
            if m["role"] == "system":
                system_instruction = m["content"]
            else:
                other_messages.append(m)

        contents = _convert_messages(other_messages)
        config = _prepare_config(response_model, **kwargs)
        if system_instruction:
            config.system_instruction = system_instruction

        buffer = io.StringIO()
        last_response = None

        # Always use streaming
        try:
            api_stream = self.client.aio.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=config,
                http_options={"timeout": effective_timeout},
            )
        except TypeError:
            # Fallback for older google-genai versions without http_options support
            api_stream = self.client.aio.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=config,
            )
        async for response in await api_stream:
            last_response = response
            if response.text:
                buffer.write(response.text)
                if streamer is not None:
                    value = (
                        ensure_json(buffer.getvalue())
                        if response_model
                        else buffer.getvalue()
                    )
                    await streamer.stream_partial(value)

        # Signal completion of streaming
        if streamer is not None:
            value = (
                ensure_json(buffer.getvalue()) if response_model else buffer.getvalue()
            )
            await streamer.stream_complete(value)

        duration = time.perf_counter() - start_time
        response_text = buffer.getvalue()
        content = (
            response_model.model_validate_json(response_text)
            if response_model
            else response_text
        )

        stats = _create_chat_stats(model_name, last_response, duration)
        return content, stats

    async def generate_image(
        self,
        model: str,
        prompt: str,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Tuple[str, ModelCallStat]:
        """
        Generate an image using Gemini models (e.g., gemini-2.5-flash-image).
        """
        start_time = time.perf_counter()
        # Handle timeout: override with method parameter or fallback to
        # self.timeout
        effective_timeout = timeout if timeout is not None else self.timeout

        model_name = get_model_name(model)

        try:
            response = await self.client.aio.models.generate_content(
                model=model_name,
                contents=[prompt],
                http_options={"timeout": effective_timeout},
                **kwargs,
            )
        except TypeError:
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

        stats = create_model_call_stat(
            call_type="image_generation",
            model=f"gemini/{model_name}",
            duration_sections=duration,
            response_data={"prompt": prompt},
        )
        return image_base64, stats

    async def compute_embeddings(
        self,
        model: str,
        texts: List[str],
        normalize: bool = False,
        normalizer: Optional[Normalizer] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Tuple[List[List[float]], ModelCallStat]:
        """Compute embeddings for a list of texts using Gemini."""
        start_time = time.perf_counter()
        # Handle timeout: override with method parameter or fallback to
        # self.timeout
        effective_timeout = timeout if timeout is not None else self.timeout

        model_name = get_model_name(model)

        try:
            response = await self.client.aio.models.embed_content(
                model=model_name,
                contents=texts,
                http_options={"timeout": effective_timeout},
                **kwargs,
            )
        except TypeError:
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

        # Estimated tokens if not provided (Gemini embed_content doesn't
        # always return usage in the same way as generate_content)
        total_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            total_tokens = response.usage_metadata.total_token_count or 0
        else:
            # Fallback: estimate tokens (rough estimation: 1 token ~= 4 chars)
            total_tokens = sum(len(t) for t in texts) // 4

        stats = create_model_call_stat(
            call_type="embedding",
            model=f"gemini/{model_name}",
            duration_sections=duration,
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
