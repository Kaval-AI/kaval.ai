import logging
import time
from datetime import datetime, timezone
from uuid import UUID
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from kavalai.agents.db import (
    LLMProfile,
    LLMCallStat,
    EmbeddingProfile,
    EmbeddingCallStat,
)
from kavalai.llm_clients.openai import OpenAIClient
from kavalai.llm_clients.gemini import GeminiClient

logger = logging.getLogger(__name__)


def get_llm_client(
    llm_profile: LLMProfile | EmbeddingProfile,
) -> OpenAIClient | GeminiClient:
    """
    Factory function to get the appropriate LLM client.
    """
    provider = llm_profile.provider.lower()
    if provider == "openai":
        return OpenAIClient(api_key=llm_profile.api_key, base_url=llm_profile.base_url)
    elif provider == "gemini":
        return GeminiClient(api_key=llm_profile.api_key)
    else:
        # Fallback for "instructor" or other strings - default to OpenAI if it looks like it
        # or raise an error. Given the requirement to still accept provider strings like instructor.
        if "openai" in provider or provider == "instructor":
            return OpenAIClient(
                api_key=llm_profile.api_key, base_url=llm_profile.base_url
            )
        elif "gemini" in provider:
            return GeminiClient(api_key=llm_profile.api_key)

        raise ValueError(f"Unsupported provider: {llm_profile.provider}")


async def _save_llm_stats(
    llm_profile: LLMProfile,
    agent_id: UUID,
    response_code: int,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    duration_seconds: float,
    cost: float,
    request_info: dict,
    response_data: any,
    session: AsyncSession = None,
):
    """
    Helper to save LLM call stats.
    """
    if not session:
        return

    try:
        stat_data = {
            "llm_profile_id": llm_profile.id,
            "agent_id": agent_id,
            "response_code": response_code,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "duration_seconds": duration_seconds,
            "cost": cost,
            "currency": "USD",
            "request_data": {"requests": [request_info]},
            "response_data": response_data,
        }

        if session.is_active:
            instance = LLMCallStat(**stat_data)
            session.add(instance)
            await session.commit()
        else:
            logger.warning("Could not save LLM call stats: Session is not active.")
    except Exception as db_err:
        # If it's the "closed transaction" error, log it as a warning
        if "closed transaction" in str(db_err):
            logger.warning(
                "Could not save LLM call stats: Session transaction is closed."
            )
        else:
            logger.error(f"Failed to save LLM call stats: {db_err}")
        try:
            await session.rollback()
        except Exception:
            pass


async def _save_embedding_stats(
    embedding_profile: EmbeddingProfile,
    agent_id: UUID,
    response_code: int,
    batch_size: int,
    total_tokens: int,
    duration_seconds: float,
    cost: float,
    request_info: dict,
    response_data: any,
    session: AsyncSession = None,
):
    """
    Helper to save embedding call stats.
    """
    if not session:
        return

    try:
        stat_data = {
            "embedding_profile_id": embedding_profile.id,
            "agent_id": agent_id,
            "response_code": response_code,
            "batch_size": batch_size,
            "total_tokens": total_tokens,
            "duration_seconds": duration_seconds,
            "cost": cost,
            "currency": "USD",
            "request_data": request_info,
            "response_data": response_data,
        }

        if session.is_active:
            instance = EmbeddingCallStat(**stat_data)
            session.add(instance)
            await session.commit()
        else:
            logger.warning(
                "Could not save embedding call stats: Session is not active."
            )
    except Exception as db_err:
        if "closed transaction" in str(db_err):
            logger.warning(
                "Could not save embedding call stats: Session transaction is closed."
            )
        else:
            logger.error(f"Failed to save embedding call stats: {db_err}")
        try:
            await session.rollback()
        except Exception:
            pass


async def chat_completion_with_stats(
    llm_profile: LLMProfile,
    response_model: type[BaseModel],
    messages: list[dict],
    session: AsyncSession = None,
    agent_id: UUID = None,
    **kwargs,
) -> any:
    """
    Execute a chat completion with native clients and collect metrics.
    """
    client = get_llm_client(llm_profile)
    start_time = time.perf_counter()

    # We want to keep track of the request for stats
    request_info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "arguments": {
            "model": llm_profile.model_name,
            "messages": messages,
            "response_model": str(response_model),
            **kwargs,
        },
    }

    try:
        result = await client.chat_completion(
            model=llm_profile.model_name,
            messages=messages,
            response_model=response_model,
            **kwargs,
        )

        duration_seconds = time.perf_counter() - start_time
        await _save_llm_stats(
            llm_profile=llm_profile,
            agent_id=agent_id,
            response_code=200,
            prompt_tokens=result["usage"]["prompt_tokens"],
            completion_tokens=result["usage"]["completion_tokens"],
            total_tokens=result["usage"]["total_tokens"],
            duration_seconds=duration_seconds,
            cost=result["cost"],
            request_info=request_info,
            response_data=result["raw_response"],
            session=session,
        )

        return result["content"]

    except Exception as e:
        duration_seconds = time.perf_counter() - start_time
        logger.error(f"Error in chat_completion_with_stats: {e}", exc_info=True)
        request_info["error"] = str(e)
        request_info["error_type"] = type(e).__name__

        await _save_llm_stats(
            llm_profile=llm_profile,
            agent_id=agent_id,
            response_code=500,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            duration_seconds=duration_seconds,
            cost=0.0,
            request_info=request_info,
            response_data={"error": str(e), "error_type": type(e).__name__},
            session=session,
        )
        raise e


async def compute_embeddings(
    llm_profile: LLMProfile | EmbeddingProfile,
    texts: list[str],
    normalize: bool = False,
    **kwargs,
) -> list[list[float]]:
    """
    Compute embeddings for a list of texts using the specified LLM profile or embedding profile.
    """
    client = get_llm_client(llm_profile)
    result = await client.compute_embeddings(
        model=llm_profile.model_name,
        texts=texts,
        normalize=normalize,
        **kwargs,
    )
    return result["embeddings"]


async def compute_embeddings_with_stats(
    llm_profile: EmbeddingProfile,
    texts: list[str],
    session: AsyncSession = None,
    agent_id: UUID = None,
    normalize: bool = False,
    **kwargs,
) -> list[list[float]]:
    """
    Compute embeddings and collect metrics.
    """
    client = get_llm_client(llm_profile)
    start_time = time.perf_counter()

    request_info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "arguments": {
            "model": llm_profile.model_name,
            "texts_count": len(texts),
            "normalize": normalize,
            **kwargs,
        },
    }

    try:
        result = await client.compute_embeddings(
            model=llm_profile.model_name,
            texts=texts,
            normalize=normalize,
            **kwargs,
        )

        duration_seconds = time.perf_counter() - start_time
        await _save_embedding_stats(
            embedding_profile=llm_profile,
            agent_id=agent_id,
            response_code=200,
            batch_size=len(texts),
            total_tokens=result["usage"]["total_tokens"],
            duration_seconds=duration_seconds,
            cost=result["cost"],
            request_info=request_info,
            response_data=result["raw_response"],
            session=session,
        )

        return result["embeddings"]

    except Exception as e:
        duration_seconds = time.perf_counter() - start_time
        logger.error(f"Error in compute_embeddings_with_stats: {e}", exc_info=True)
        request_info["error"] = str(e)
        request_info["error_type"] = type(e).__name__

        await _save_embedding_stats(
            embedding_profile=llm_profile,
            agent_id=agent_id,
            response_code=500,
            batch_size=len(texts),
            total_tokens=0,
            duration_seconds=duration_seconds,
            cost=0.0,
            request_info=request_info,
            response_data={"error": str(e), "error_type": type(e).__name__},
            session=session,
        )
        raise e
