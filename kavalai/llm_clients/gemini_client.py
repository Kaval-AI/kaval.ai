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
import os
import time
from typing import Any, Dict, List, Optional, Type, Tuple

from google import genai
from google.genai import types
from partial_json_parser import ensure_json
from pydantic import BaseModel

from kavalai.agents.db import ModelCallStat
from kavalai.llm_clients.common import (
    create_model_call_stat,
    get_model_name,
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

    def _convert_messages(
        self, messages: List[Dict[str, Any]]
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
                    elif item.get("type") == "image_url":
                        image_url = item.get("image_url", {}).get("url", "")
                        if image_url.startswith("data:image/"):
                            header, encoded = image_url.split(",", 1)
                            mime_type = header.split(";")[0].split(":")[1]
                            image_bytes = base64.b64decode(encoded)
                            parts.append(
                                types.Part.from_bytes(
                                    data=image_bytes, mime_type=mime_type
                                )
                            )

            contents.append(types.Content(role=gemini_role, parts=parts))

        return system_instruction, contents

    async def chat_completions(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        response_model: Optional[Type[BaseModel]] = None,
        streamer: Optional[Streamer] = None,
        reasoning_effort: Optional[str] = None,
        thinking_level: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Any, ModelCallStat]:
        start_time = time.perf_counter()
        model_name = get_model_name(model)

        system_instruction, contents = self._convert_messages(messages)

        config_kwargs = {**kwargs}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        if response_model:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_model

        # Support for reasoning/thinking (e.g. for Gemini 2.0 Flash Thinking)
        if thinking_budget is not None:
            config_kwargs["thinking_config"] = {
                "include_thoughts": True,
                "include_thoughts_in_response": True,
            }
            # Note: actual thinking budget param name might vary or be part of thinking_config

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
                            buffer.write(part.text)
                            if streamer is not None:
                                value = (
                                    ensure_json(buffer.getvalue())
                                    if response_model
                                    else buffer.getvalue()
                                )
                                await streamer.stream_partial(value)
                        if part.thought:
                            thought_buffer.write(part.text)

            if chunk.usage_metadata:
                input_tokens = chunk.usage_metadata.prompt_token_count
                output_tokens = chunk.usage_metadata.candidates_token_count

        result_text = buffer.getvalue()
        thought_text = thought_buffer.getvalue()

        if streamer is not None:
            value = ensure_json(result_text) if response_model else result_text
            await streamer.stream_complete(value)

        if response_model:
            result = response_model.model_validate_json(result_text)
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
            response_data=response_data,
        )
        return result, stats

    async def generate_image(
        self,
        model: str,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Tuple[Optional[str], ModelCallStat]:
        start_time = time.perf_counter()
        model_name = get_model_name(model)

        try:
            response = await self.client.aio.models.generate_images(
                model=model_name,
                prompt=prompt,
                config=types.GenerateImagesConfig(number_of_images=1, **kwargs),
            )

            image_base64 = None
            if response.generated_images:
                image_bytes = response.generated_images[0].image.image_bytes
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        except Exception:
            image_base64 = None

        duration = time.perf_counter() - start_time

        stats = create_model_call_stat(
            call_type="image_generation",
            model=f"gemini/{model_name}",
            duration_sections=duration,
            response_data={"size": size, "quality": quality},
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
            response_data=None,
        )
        return embeddings, stats

    async def list_models(self) -> List[str]:
        models = []
        for m in await self.client.aio.models.list():
            models.append(m.name)
        return models
