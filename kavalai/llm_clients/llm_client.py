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

from loguru import logger
import os
import asyncio
import random
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar, Optional

from pydantic import BaseModel
import openai
from google.genai import errors

from kavalai.agents.db import ModelCallStat
from kavalai.normalizer import Normalizer
from kavalai.llm_clients.gemini_client import GeminiClient
from kavalai.llm_clients.openai_client import OpenAIClient
from kavalai.llm_clients.ollama_client import OllamaClient
from kavalai.llm_clients.common import Streamer
from kavalai.llm_clients.kwargs_mapper import LLMKWargsMapper


T = TypeVar("T")


async def with_retry(
    func: Callable,
    *args,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    **kwargs,
) -> Any:
    """
    Exponential backoff retry wrapper for LLM client calls.
    Retries only on specific OpenAI and Gemini exceptions.

    :param func: The function to call.
    :param max_retries: Maximum number of retries.
    :param base_delay: Initial delay in seconds.
    :param max_delay: Maximum delay in seconds.
    :return: The result of the function call.
    """
    retriable_exceptions = (
        # OpenAI exceptions
        openai.RateLimitError,
        openai.InternalServerError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.LengthFinishReasonError,
        # Gemini exceptions
        errors.ServerError,
        errors.ClientError,
    )

    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except errors.ClientError as e:
            # Special handling for Gemini ClientError to avoid retrying on 404
            if hasattr(e, "status") and e.status == 404:
                raise e
            if "404" in str(e):
                raise e
            last_exception = e
        except retriable_exceptions as e:
            last_exception = e
            if attempt == max_retries:
                break

            delay = min(base_delay * (2**attempt) + random.uniform(0, 1), max_delay)
            logger.warning(
                f"LLM call to {args[0] if args else 'unknown'} failed with {type(e).__name__}: {str(e)}. "
                f"Retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries})..."
            )
            await asyncio.sleep(delay)
        except Exception as e:
            # Do not retry on other exceptions (programming errors, auth errors, etc.)
            raise e

    raise last_exception


class LLMClient:
    """
    Unified LLM client that handles provider selection, retries, and statistics.
    """

    def __init__(self, model: str):
        self.full_model = model
        self.provider, self.model_name = model.split("/")
        self.client = self._get_underlying_client()

    def _get_underlying_client(self) -> OpenAIClient | GeminiClient | OllamaClient:
        """
        Factory method to get the appropriate LLM client.
        """
        timeout = float(os.environ.get("KAVALAI_LLM_TIMEOUT", 30.0))
        if self.provider == "openai":
            return OpenAIClient(
                api_key=os.environ["OPENAI_API_KEY"],
                service_tier=os.environ.get("KAVALAI_OPENAI_SERVICE_TIER"),
                timeout=timeout,
            )
        elif self.provider == "gemini":
            return GeminiClient(
                api_key=os.environ["GEMINI_API_KEY"],
                timeout=timeout,
            )
        elif self.provider == "ollama":
            return OllamaClient(
                host=os.environ.get("OLLAMA_HOST"),
                timeout=timeout,
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
