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

import os
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel

from kavalai.agents.db import ModelCallStat
from kavalai.llm_clients.common import Streamer
from kavalai.llm_clients.fastembed_client import FastEmbedClient
from kavalai.llm_clients.gemini_client import GeminiClient
from kavalai.llm_clients.kwargs_mapper import LLMKWargsMapper
from kavalai.llm_clients.ollama_client import OllamaClient
from kavalai.llm_clients.openai_client import OpenAIClient
from kavalai.llm_clients.with_retry import with_retry
from kavalai.normalizer import Normalizer


class LLMClient:
    """
    Unified LLM client that handles provider selection, retries, and statistics.
    """

    def __init__(self, model: str):
        self.full_model = model
        self.provider, self.model_name = model.split("/", maxsplit=1)
        self.client = self._get_underlying_client()

    def _get_underlying_client(
        self,
    ) -> OpenAIClient | GeminiClient | OllamaClient | FastEmbedClient:
        """
        Factory method to get the appropriate LLM client.
        """
        timeout = float(os.environ.get("KAVALAI_LLM_TIMEOUT", 30.0))
        if self.provider == "openai":
            return OpenAIClient(
                api_key=os.environ.get("OPENAI_API_KEY"),
                service_tier=os.environ.get("KAVALAI_OPENAI_SERVICE_TIER"),
                timeout=timeout,
            )
        elif self.provider == "gemini":
            return GeminiClient(
                api_key=os.environ.get("GEMINI_API_KEY"),
                timeout=timeout,
            )
        elif self.provider == "ollama":
            return OllamaClient(
                host=os.environ.get("OLLAMA_HOST"),
                timeout=timeout,
            )
        elif self.provider == "fastembed":
            return FastEmbedClient(
                cache_dir=os.environ.get("FASTEMBED_CACHE_DIR"),
                threads=int(os.environ.get("FASTEMBED_THREADS"))
                if os.environ.get("FASTEMBED_THREADS")
                else None,
            )
        else:
            raise ValueError(f"Invalid provider: {self.provider}")

    async def chat_completions(
        self,
        messages: list[dict],
        response_model: type[BaseModel] | None = None,
        streamer: Streamer | None = None,
        thinking_budget: Optional[int] = None,
        stream_delta: bool = False,
        **kwargs,
    ) -> tuple[Any, ModelCallStat]:
        """
        Execute a chat completion with native clients and return result and stats.
        """
        # Ensure explicitly provided named params also flow through kwargs mapping
        # so provider-specific clients can receive them.
        if thinking_budget is not None:
            kwargs = dict(kwargs)
            kwargs["thinking_budget"] = thinking_budget

        # We want to keep track of the request for stats
        request_info = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "arguments": {
                "model": self.full_model,
                "messages": messages,
                "response_model": str(response_model) if response_model else None,
                "streamer": str(streamer) if streamer else None,
                "thinking_budget": thinking_budget,
                "stream_delta": stream_delta,
                **kwargs,
            },
        }

        # Map user-friendly/common kwargs to provider-specific ones
        mapped_kwargs = LLMKWargsMapper.map(self.provider, self.model_name, kwargs)

        # Ensure 'model' is not twice in kwargs
        mapped_kwargs.pop("model", None)

        content, stats = await with_retry(
            self.client.chat_completions,
            model=self.model_name,
            messages=messages,
            response_model=response_model,
            streamer=streamer,
            stream_delta=stream_delta,
            **mapped_kwargs,
        )

        stats.request_data = {"requests": [request_info]}

        return content, stats

    async def list_models(self) -> list[str]:
        return await self.client.list_models()

    async def compute_embeddings(
        self,
        texts: list[str],
        normalize: bool = False,
        normalizer: Normalizer | None = None,
        **kwargs,
    ) -> tuple[list[list[float]], ModelCallStat]:
        """
        Compute embeddings for a list of texts and return embeddings and stats.
        """
        request_info = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "arguments": {
                "model": self.full_model,
                "texts_count": len(texts),
                "normalize": normalize,
                **kwargs,
            },
        }

        # Ensure 'model' is not twice in kwargs
        kwargs.pop("model", None)

        embeddings, stats = await with_retry(
            self.client.compute_embeddings,
            model=self.model_name,
            texts=texts,
            normalize=normalize,
            normalizer=normalizer,
            **kwargs,
        )

        stats.request_data = request_info

        return embeddings, stats
