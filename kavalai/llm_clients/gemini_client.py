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

import io
import time
from typing import Any, Dict, List, Optional, Type, Tuple

from google import genai
from google.genai import types
from partial_json_parser import ensure_json
from pydantic import BaseModel

from kavalai.agents.db import ModelCallStat
from kavalai.normalizer import Normalizer, get_default_normalizer
from kavalai.llm_clients.common import (
    create_model_call_stat,
    Streamer,
)


class GeminiClient:
    def __init__(self, api_key: str, timeout: float = 30.0):
        self.client = genai.Client(api_key=api_key, http_options={"timeout": timeout})

    async def chat_completions(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        response_model: Type[BaseModel] = None,
        streamer: Optional[Streamer] = None,
        **kwargs,
    ) -> Tuple[Any, ModelCallStat]:
        start_time = time.perf_counter()

        # Convert messages to Gemini format
        contents = self._convert_messages(messages)

        config_kwargs = {}
        schema = response_model.model_json_schema()
        self._cleanup_schema(schema)
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = schema

        # Extract other kwargs for GenerateContentConfig
        for key in [
            "temperature",
            "top_p",
            "top_k",
            "max_output_tokens",
            "stop_sequences",
        ]:
            if key in kwargs:
                config_kwargs[key] = kwargs.pop(key)

        config = types.GenerateContentConfig(**config_kwargs)

        buffer = io.StringIO()
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        last_response = None

        # Using async call
        async for response in await self.client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        ):
            last_response = response
            if response.text:
                buffer.write(response.text)
                if streamer is not None:
                    await streamer.stream_partial(ensure_json(buffer.getvalue()))
        # Stream the final complete value.
        if streamer is not None:
            await streamer.stream_complete(ensure_json(buffer.getvalue()))
        duration = time.perf_counter() - start_time

        content = response_model.model_validate_json(buffer.getvalue())

        if last_response and last_response.usage_metadata:
            usage_metadata = last_response.usage_metadata
            prompt_tokens = usage_metadata.prompt_token_count or 0
            completion_tokens = usage_metadata.candidates_token_count or 0
            total_tokens = usage_metadata.total_token_count or 0

        stats = create_model_call_stat(
            call_type="llm",
            model=f"gemini/{model}",
            duration=duration,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            response_data=str(last_response) if last_response else None,
        )

        return content, stats

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[types.Content]:
        gemini_messages = []
        for m in messages:
            role = m["role"]
            if role == "assistant":
                role = "model"
            gemini_messages.append(
                types.Content(role=role, parts=[types.Part(text=m["content"])])
            )
        return gemini_messages

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

    async def compute_embeddings(
        self,
        model: str,
        texts: List[str],
        normalize: bool = False,
        normalizer: Optional[Normalizer] = None,
        **kwargs,
    ) -> Tuple[List[List[float]], ModelCallStat]:
        start_time = time.perf_counter()
        response = await self.client.aio.models.embed_content(
            model=model,
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
        # However, let's check if it exists
        total_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            total_tokens = response.usage_metadata.total_token_count or 0
        else:
            # Fallback: estimate tokens (rough estimation: 1 token ~= 4 chars)
            total_tokens = sum(len(t) for t in texts) // 4

        stats = create_model_call_stat(
            call_type="embedding",
            model=f"gemini/{model}",
            duration=duration,
            batch_size=len(texts),
            total_tokens=total_tokens,
            response_data=str(response),
        )

        return embeddings, stats
