import logging
import os
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from kavalai.agents.db import ModelCallStat
from kavalai.llm_clients.gemini import GeminiClient
from kavalai.llm_clients.openai import OpenAIClient

logger = logging.getLogger(__name__)


def get_llm_client(
    model: str,
) -> OpenAIClient | GeminiClient:
    """
    Factory function to get the appropriate LLM client.

    Model parameter is a provider/model combo like openai/text-embedding-3-small
    """
    provider, model = model.split("/")
    if provider == "openai":
        return OpenAIClient(api_key=os.environ["OPENAI_API_KEY"])
    elif provider == "gemini":
        return GeminiClient(api_key=os.environ["GEMINI_API_KEY"])
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
    content, stats = await client.chat_completion(
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
    embeddings, stats = await client.compute_embeddings(
        model=model_name,
        texts=texts,
        normalize=normalize,
        **kwargs,
    )

    stats.request_data = request_info

    return embeddings, stats
