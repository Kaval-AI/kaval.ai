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

import math
import time
from typing import Any, Dict, List, Optional, Type, Tuple

from openai import AsyncOpenAI
from pydantic import BaseModel

from kavalai.agents.db import ModelCallStat


class OpenAIClient:
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        service_tier: Optional[str] = None,
    ):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.service_tier = service_tier
        assert service_tier in ["standard", "priority", None]

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        response_model: Type[BaseModel],
        **kwargs,
    ) -> Tuple[Any, ModelCallStat]:
        start_time = time.perf_counter()

        call_kwargs = {
            "model": model,
            "messages": messages,
            "response_format": response_model,
            **kwargs,
        }
        if self.service_tier:
            call_kwargs["service_tier"] = self.service_tier

        response = await self.client.beta.chat.completions.parse(**call_kwargs)
        content = response.choices[0].message.parsed

        duration = time.perf_counter() - start_time

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

        stats = ModelCallStat(
            call_type="llm",
            model=f"openai/{model}",
            response_code=200,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            duration_seconds=duration,
            cost=None,  # We will compute cost later
            response_data=response.model_dump()
            if hasattr(response, "model_dump")
            else str(response),
        )

        return content, stats

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
        if self.service_tier:
            call_kwargs["service_tier"] = self.service_tier

        response = await self.client.embeddings.create(**call_kwargs)
        duration = time.perf_counter() - start_time

        embeddings = [data.embedding for data in response.data]

        if normalize:
            normalized_embeddings = []
            for emb in embeddings:
                norm = math.sqrt(sum(x * x for x in emb))
                if norm > 0:
                    normalized_embeddings.append([x / norm for x in emb])
                else:
                    normalized_embeddings.append(emb)
            embeddings = normalized_embeddings

        total_tokens = response.usage.total_tokens if response.usage else 0

        stats = ModelCallStat(
            call_type="embedding",
            model=f"openai/{model}",
            response_code=200,
            batch_size=len(texts),
            total_tokens=total_tokens,
            duration_seconds=duration,
            response_data=response.model_dump()
            if hasattr(response, "model_dump")
            else str(response),
        )

        return embeddings, stats
