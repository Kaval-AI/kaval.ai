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

from openai import AsyncOpenAI
from partial_json_parser import ensure_json
from pydantic import BaseModel

from kavalai.agents.db import ModelCallStat
from kavalai.normalizer import Normalizer, get_default_normalizer
from kavalai.llm_clients.common import (
    create_model_call_stat,
    Streamer,
)


class OpenAIClient:
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        service_tier: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.timeout = timeout
        self.client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=self.timeout
        )
        self.service_tier = service_tier
        assert service_tier in ["auto", "default", "flex", "priority", None]

    async def chat_completions(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        response_model: Optional[Type[BaseModel]] = None,
        streamer: Optional[Streamer] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Tuple[Any, ModelCallStat]:
        start_time = time.perf_counter()

        # Handle timeout: override with method parameter or fallback to self.timeout
        effective_timeout = timeout if timeout is not None else self.timeout

        formatted_messages = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                # Message already in content-list format
                formatted_messages.append(m)
            elif m.get("images"):
                # Handle images provided in a separate field
                msg_content = [{"type": "text", "text": content}]
                for img_base64 in m["images"]:
                    msg_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            },
                        }
                    )
                formatted_messages.append({"role": m["role"], "content": msg_content})
            else:
                formatted_messages.append(m)

        call_kwargs = {
            "model": model,
            "messages": formatted_messages,
            "timeout": effective_timeout,
            **kwargs,
        }
        if response_model:
            call_kwargs["response_format"] = response_model
        if self.service_tier:
            call_kwargs["service_tier"] = self.service_tier

        buffer = io.StringIO()
        async with self.client.beta.chat.completions.stream(**call_kwargs) as stream:
            async for chunk in stream:
                if chunk.type == "content.delta":
                    buffer.write(chunk.delta)
                    if streamer is not None and buffer.getvalue().strip():
                        await streamer.stream_partial(ensure_json(buffer.getvalue()))

            final_completion = await stream.get_final_completion()
            usage = final_completion.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
        # Stream the final complete value.
        if streamer is not None:
            await streamer.stream_complete(buffer.getvalue())

        result = buffer.getvalue()
        if response_model:
            result = response_model.model_validate_json(result)

        duration = time.perf_counter() - start_time

        stats = create_model_call_stat(
            call_type="llm",
            model=f"openai/{model}",
            duration=duration,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            response_data=result.model_dump()
            if hasattr(result, "model_dump")
            else result,
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
    ) -> Tuple[str, ModelCallStat]:
        start_time = time.perf_counter()
        # Handle timeout: override with method parameter or fallback to self.timeout
        effective_timeout = timeout if timeout is not None else self.timeout

        # Ensure response_format is always b64_json
        kwargs.pop("response_format", None)
        # Normalize quality: OpenAI API supports 'standard' and 'hd' for DALL-E 3
        if quality not in {"standard", "hd"}:
            quality = "standard"
        response = await self.client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            timeout=effective_timeout,
            **kwargs,
        )
        duration = time.perf_counter() - start_time
        # Prefer base64 payload if available; otherwise fall back to URL
        data0 = response.data[0]
        image_base64 = getattr(data0, "b64_json", None)
        if image_base64 is None and hasattr(data0, "url") and data0.url:
            # Fetch the URL and convert to base64 to keep a consistent return type
            import base64
            import httpx

            with httpx.Client(timeout=60.0) as client:
                resp = client.get(data0.url)
                resp.raise_for_status()
                image_base64 = base64.b64encode(resp.content).decode("utf-8")

        stats = create_model_call_stat(
            call_type="image_generation",
            model=f"openai/{model}",
            duration=duration,
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
        # Handle timeout: override with method parameter or fallback to self.timeout
        effective_timeout = timeout if timeout is not None else self.timeout

        call_kwargs = {
            "input": texts,
            "model": model,
            "timeout": effective_timeout,
            **kwargs,
        }
        response = await self.client.embeddings.create(**call_kwargs)
        duration = time.perf_counter() - start_time

        embeddings = [data.embedding for data in response.data]
        if normalize:
            if normalizer is None:
                normalizer = get_default_normalizer()
            embeddings = normalizer.transform(embeddings)

        total_tokens = response.usage.total_tokens if response.usage else 0

        stats = create_model_call_stat(
            call_type="embedding",
            model=f"openai/{model}",
            duration=duration,
            batch_size=len(texts),
            total_tokens=total_tokens,
            response_data=response.model_dump()
            if hasattr(response, "model_dump")
            else response,
        )
        return embeddings, stats

    async def list_models(self) -> List[str]:
        response = await self.client.models.list()
        return [model.id for model in response.data]
