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

import logging
import os
import asyncio
import random
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

from pydantic import BaseModel
import openai
from google.genai import errors

from kavalai.agents.db import ModelCallStat
from kavalai.llm_clients.gemini import GeminiClient
from kavalai.llm_clients.openai import OpenAIClient

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def with_retry(
    func: Callable[..., T],
    *args,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    **kwargs,
) -> T:
    """
    Exponential backoff retry wrapper for LLM client calls.
    Retries only on specific OpenAI and Gemini exceptions.
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
        except retriable_exceptions as e:
            last_exception = e
            if attempt == max_retries:
                break

            delay = min(base_delay * (2**attempt) + random.uniform(0, 1), max_delay)
            logger.warning(
                f"LLM call failed with {type(e).__name__}: {e}. "
                f"Retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries})..."
            )
            await asyncio.sleep(delay)
        except Exception as e:
            # Do not retry on other exceptions (programming errors, auth errors, etc.)
            raise e

    raise last_exception


def get_llm_client(
    model: str,
) -> OpenAIClient | GeminiClient:
    """
    Factory function to get the appropriate LLM client.

    Model parameter is a provider/model combo like openai/text-embedding-3-small
    """
    timeout = float(os.environ.get("KAVALAI_LLM_TIMEOUT", 30.0))
    provider, model_name = model.split("/")
    if provider == "openai":
        return OpenAIClient(
            api_key=os.environ["OPENAI_API_KEY"],
            service_tier=os.environ.get("KAVALAI_OPENAI_SERVICE_TIER"),
            timeout=timeout,
        )
    elif provider == "gemini":
        return GeminiClient(api_key=os.environ["GEMINI_API_KEY"], timeout=timeout)
    else:
        raise ValueError(f"Invalid provider: {provider}")


async def chat_completions(
    model: str,
    response_model: type[BaseModel],
    messages: list[dict],
    **kwargs,
) -> tuple[Any, ModelCallStat]:
    """
    Execute a chat completion with native clients and return result and stats.
    """
    client = get_llm_client(model)

    # We want to keep track of the request for stats
    request_info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "arguments": {
            "model": model,
            "messages": messages,
            "response_model": str(response_model),
            **kwargs,
        },
    }
    _, model_name = model.split("/")

    content, stats = await with_retry(
        client.chat_completion,
        model=model_name,
        messages=messages,
        response_model=response_model,
        **kwargs,
    )

    stats.request_data = {"requests": [request_info]}

    return content, stats


async def compute_embeddings(
    model: str,
    texts: list[str],
    normalize: bool = False,
    **kwargs,
) -> tuple[list[list[float]], ModelCallStat]:
    """
    Compute embeddings for a list of texts and return embeddings and stats.
    """
    client = get_llm_client(model)

    request_info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "arguments": {
            "model": model,
            "texts_count": len(texts),
            "normalize": normalize,
            **kwargs,
        },
    }
    _, model_name = model.split("/")

    embeddings, stats = await with_retry(
        client.compute_embeddings,
        model=model_name,
        texts=texts,
        normalize=normalize,
        **kwargs,
    )

    stats.request_data = request_info

    return embeddings, stats
