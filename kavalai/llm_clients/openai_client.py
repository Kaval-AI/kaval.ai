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
from openai.types.responses import ResponseTextDeltaEvent, ResponseCompletedEvent
from partial_json_parser import ensure_json
from pydantic import BaseModel

from kavalai.agents.db import ModelCallStat
from kavalai.llm_clients.common import (
    create_model_call_stat,
    normalize_embeddings,
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
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.service_tier = service_tier
        assert service_tier in ["auto", "default", "flex", "priority", None]

    async def chat_completions(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        response_model: Type[BaseModel],
        streamer: Optional[Streamer] = None,
        **kwargs,
    ) -> Tuple[Any, ModelCallStat]:
        start_time = time.perf_counter()

        call_kwargs = {
            "model": model,
            "input": messages,
            "text_format": response_model,
            **kwargs,
        }
        if self.service_tier:
            call_kwargs["service_tier"] = self.service_tier

        buffer = io.StringIO()
        async with self.client.responses.stream(**call_kwargs) as stream:
            async for chunk in stream:
                if isinstance(chunk, ResponseTextDeltaEvent):
                    buffer.write(chunk.delta)
                    if streamer is not None:
                        await streamer.stream_partial(ensure_json(buffer.getvalue()))
                if isinstance(chunk, ResponseCompletedEvent):
                    usage = chunk.response.usage
                    input_tokens = usage.input_tokens
                    output_tokens = usage.output_tokens
        # Stream the final complete value.
        if streamer is not None:
            await streamer.stream_complete(buffer.getvalue())

        result = response_model.model_validate_json(buffer.getvalue())
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

    async def compute_embeddings(
        self,
        model: str,
        texts: List[str],
        normalize: bool = False,
        **kwargs,
    ) -> Tuple[List[List[float]], ModelCallStat]:
        start_time = time.perf_counter()
        call_kwargs = {
            "input": texts,
            "model": model,
            **kwargs,
        }
        response = await self.client.embeddings.create(**call_kwargs)
        duration = time.perf_counter() - start_time

        embeddings = [data.embedding for data in response.data]

        if normalize:
            embeddings = normalize_embeddings(embeddings)

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
