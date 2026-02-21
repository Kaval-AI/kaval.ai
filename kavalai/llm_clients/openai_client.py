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
import os
import time
from typing import Any, Dict, List, Optional, Type, Tuple

from openai import AsyncOpenAI
from openai.types.responses import (
    ResponseTextDeltaEvent,
    ResponseRefusalDeltaEvent,
    ResponseErrorEvent,
    ResponseCompletedEvent,
)
from partial_json_parser import ensure_json
from pydantic import BaseModel

from kavalai.agents.db import ModelCallStat
from kavalai.llm_clients.common import (
    create_model_call_stat,
    Streamer,
)
from kavalai.normalizer import Normalizer, get_default_normalizer


class OpenAIClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        service_tier: Optional[str] = None,
        timeout: float = 30.0,
    ):
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
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
        **kwargs,
    ) -> Tuple[Any, ModelCallStat]:
        start_time = time.perf_counter()

        call_kwargs = {
            "model": model,
            "input": messages,
            **kwargs,
        }
        if response_model and issubclass(response_model, BaseModel):
            call_kwargs["text_format"] = response_model
        elif response_model:
            raise ValueError("response_model must be a pydantic BaseModel")
        if self.service_tier:
            call_kwargs["service_tier"] = self.service_tier
        buffer = io.StringIO()
        async with self.client.responses.stream(**call_kwargs) as stream:
            async for event in stream:
                if isinstance(event, ResponseTextDeltaEvent):
                    buffer.write(event.delta)
                    if streamer is not None:
                        value = (
                            ensure_json(buffer.getvalue())
                            if response_model
                            else buffer.getvalue()
                        )
                        await streamer.stream_partial(value)
                elif isinstance(event, ResponseRefusalDeltaEvent):
                    buffer.write(event.delta)
                    if streamer is not None:
                        value = (
                            ensure_json(buffer.getvalue())
                            if response_model
                            else buffer.getvalue()
                        )
                        await streamer.stream_partial(value)
                elif isinstance(event, ResponseErrorEvent):
                    raise RuntimeError(event.error)
                elif isinstance(event, ResponseCompletedEvent):
                    usage = event.response.usage
                    input_tokens = usage.input_tokens
                    output_tokens = usage.output_tokens
        # Stream the final complete value.
        if streamer is not None:
            value = (
                ensure_json(buffer.getvalue()) if response_model else buffer.getvalue()
            )
            await streamer.stream_complete(value)
        result = buffer.getvalue()
        if response_model:
            result = response_model.model_validate_json(result)

        duration = time.perf_counter() - start_time

        stats = create_model_call_stat(
            call_type="llm",
            model=f"openai/{model}",
            duration_sections=duration,
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
        # Handle timeout: override with method parameter or fallback to
        # self.timeout
        effective_timeout = timeout if timeout is not None else self.timeout

        # Ensure tool type is image_generation if using the new responses API
        response = await self.client.responses.create(
            model=model,
            input=prompt,
            tools=[{"type": "image_generation"}],
            timeout=effective_timeout,
            **kwargs,
        )

        duration = time.perf_counter() - start_time

        # Save the image to a file
        image_data = [
            output.result
            for output in response.output
            if output.type == "image_generation_call"
        ]

        image_base64 = None
        if image_data:
            image_base64 = image_data[0]

        stats = create_model_call_stat(
            call_type="image_generation",
            model=f"openai/{model}",
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
        # Handle timeout: override with method parameter or fallback to
        # self.timeout
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
            duration_sections=duration,
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
