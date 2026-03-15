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
import json
import os
import time
from typing import Any, Dict, List, Optional, Type, Tuple

from google import genai
from google.genai import types
from pydantic import BaseModel

from kavalai.agents.db import ModelCallStat
from kavalai.llm_clients.common import (
    create_model_call_stat,
    get_model_name,
    fix_json,
    Streamer,
)
from kavalai.normalizer import Normalizer, get_default_normalizer


class GeminiClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
        self.timeout = timeout
        self.client = genai.Client(api_key=api_key)

    async def chat_completions(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        response_model: Optional[Type[BaseModel]] = None,
        streamer: Optional[Streamer] = None,
        thinking_budget: Optional[int] = None,
        stream_delta: bool = False,
        **kwargs,
    ) -> Tuple[Any, ModelCallStat]:
        start_time = time.perf_counter()
        model_name = get_model_name(model)

        system_instruction, contents = convert_messages(messages)

        config_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        if response_model:
            config_kwargs["response_mime_type"] = "application/json"
            # Gemini doesn't support additional_properties in schema
            schema = response_model.model_json_schema()
            remove_additional_properties(schema)
            config_kwargs["response_schema"] = schema

        # Support for reasoning/thinking (e.g. for Gemini 2.0 Flash Thinking)
        if thinking_budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                include_thoughts=True,
            )
        config = types.GenerateContentConfig(**config_kwargs)

        buffer = io.StringIO()
        thought_buffer = io.StringIO()
        input_tokens = 0
        output_tokens = 0

        # We use streaming by default to unify the implementation
        async for chunk in await self.client.aio.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=config,
        ):
            if chunk.candidates:
                candidate = chunk.candidates[0]
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.text:
                            if streamer is not None:
                                if stream_delta:
                                    await streamer.stream_partial(part.text)
                                    buffer.write(part.text)
                                elif response_model:
                                    # For structured output, we still want to stream the full partial JSON
                                    # to allow the UI to parse it.
                                    buffer.write(part.text)
                                    # fix_json now returns a dict/list, so we convert back to JSON string for streaming
                                    value = fix_json(buffer.getvalue())
                                    await streamer.stream_partial(json.dumps(value))
                                else:
                                    await streamer.stream_partial(part.text)
                                    buffer.write(part.text)
                            else:
                                buffer.write(part.text)
                        if part.thought:
                            thought_buffer.write(part.text)
                            if streamer is not None:
                                if stream_delta:
                                    await streamer.stream_partial(
                                        part.text, name=f"{streamer.name}_thought"
                                    )
                                else:
                                    await streamer.stream_partial(
                                        thought_buffer.getvalue(),
                                        name=f"{streamer.name}_thought",
                                    )

            if chunk.usage_metadata:
                input_tokens = chunk.usage_metadata.prompt_token_count
                output_tokens = chunk.usage_metadata.candidates_token_count

        result_text = buffer.getvalue()
        thought_text = thought_buffer.getvalue()

        if streamer is not None:
            value = fix_json(result_text) if response_model else result_text
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            await streamer.stream_complete(value)

        if response_model:
            result = response_model.model_validate(fix_json(result_text))
        else:
            result = result_text

        duration = time.perf_counter() - start_time

        response_data = result.model_dump() if hasattr(result, "model_dump") else result
        if thought_text:
            if isinstance(response_data, dict):
                response_data["thought"] = thought_text
            else:
                response_data = {"result": response_data, "thought": thought_text}

        stats = create_model_call_stat(
            call_type="llm",
            model=f"gemini/{model_name}",
            duration_sections=duration,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            cost=None,
            response_data=response_data,
        )
        stats.currency = None
        return result, stats

    async def compute_embeddings(
        self,
        model: str,
        texts: List[str],
        normalize: bool = False,
        normalizer: Optional[Normalizer] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Tuple[List[List[float]], ModelCallStat]:
        start_time = time.perf_counter()
        model_name = get_model_name(model)

        response = await self.client.aio.models.embed_content(
            model=model_name, contents=texts, config=types.EmbedContentConfig(**kwargs)
        )

        duration = time.perf_counter() - start_time

        embeddings = [embedding.values for embedding in response.embeddings]

        if normalize:
            if normalizer is None:
                normalizer = get_default_normalizer()
            embeddings = normalizer.transform(embeddings)

        total_tokens = 0

        stats = create_model_call_stat(
            call_type="embedding",
            model=f"gemini/{model_name}",
            duration_sections=duration,
            batch_size=len(texts),
            total_tokens=total_tokens,
            cost=None,
            response_data=None,
        )
        stats.currency = None
        return embeddings, stats

    async def list_models(self) -> List[str]:
        models = []
        for m in await self.client.aio.models.list():
            models.append(m.name)
        return models


def convert_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[Optional[str], List[types.Content]]:
    system_instruction = None
    contents = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            system_instruction = content
            continue

        # Convert role to Gemini format (user or model)
        gemini_role = "user" if role == "user" else "model"

        parts = []
        if isinstance(content, str):
            parts.append(types.Part.from_text(text=content))
        elif isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    parts.append(types.Part.from_text(text=item.get("text")))

        contents.append(types.Content(role=gemini_role, parts=parts))

    return system_instruction, contents


def remove_additional_properties(schema: Dict[str, Any]) -> None:
    """
    Recursively remove 'additionalProperties' from a JSON schema.
    Gemini's API doesn't support this field.
    """
    if not isinstance(schema, dict):
        return

    # Remove additionalProperties if present
    schema.pop("additionalProperties", None)

    # Recursively process nested objects
    if "properties" in schema:
        for prop_schema in schema["properties"].values():
            remove_additional_properties(prop_schema)

    # Handle arrays
    if "items" in schema:
        remove_additional_properties(schema["items"])

    # Handle allOf, anyOf, oneOf
    for key in ["allOf", "anyOf", "oneOf"]:
        if key in schema:
            for sub_schema in schema[key]:
                remove_additional_properties(sub_schema)

    # Handle $defs or definitions (where nested models are stored)
    for key in ["$defs", "definitions"]:
        if key in schema:
            for def_schema in schema[key].values():
                remove_additional_properties(def_schema)
