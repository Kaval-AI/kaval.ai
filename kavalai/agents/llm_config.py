import logging
import os
from datetime import datetime, timezone
from uuid import UUID

import time
import yaml
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from kavalai import crud
from kavalai.agents.db import LLMProfile, LLMCallStat
from kavalai.llm_clients.openai import OpenAIClient
from kavalai.llm_clients.gemini import GeminiClient

logger = logging.getLogger(__name__)


class LLMProfileView(BaseModel):
    """LLM profile data without API key."""

    id: UUID
    name: str
    provider: str
    model_name: str
    base_url: str | None
    default_mode: str | None
    created_at: datetime
    updated_at: datetime


async def get_llm_profiles_from_db(
    session: AsyncSession,
) -> list[LLMProfileView]:
    profiles = await crud.get_all(session, LLMProfile)
    # Sort by updated_at desc manually since crud.get_all doesn't support ordering yet
    profiles = sorted(profiles, key=lambda p: p.updated_at, reverse=True)

    return [
        LLMProfileView(
            id=p.id,
            name=p.name,
            provider=p.provider,
            model_name=p.model_name,
            base_url=p.base_url,
            default_mode=p.default_mode,
            created_at=p.created_at,
            updated_at=p.updated_at,
        )
        for p in profiles
    ]


def load_profile_from_path(
    profile_name: str, folder_path: str = None
) -> LLMProfile | None:
    if folder_path is None:
        folder_path = os.getenv("LLM_PROFILES_PATH", "llm_profiles")
    profile_path = os.path.join(folder_path, f"{profile_name}.yaml")
    if not os.path.exists(profile_path):
        return None

    try:
        with open(profile_path, "r") as f:
            data = yaml.safe_load(f)
            return LLMProfile(**data)
    except Exception as e:
        logger.error(
            f"Error loading LLM profile '{profile_name}' from {profile_path}: {e}"
        )
        return None


def get_llm_client(llm_profile: LLMProfile) -> OpenAIClient | GeminiClient:
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


async def chat_completion_with_stats(
    llm_profile: LLMProfile,
    response_model: type[BaseModel],
    messages: list[dict],
    session: AsyncSession = None,
    **kwargs,
) -> any:
    """
    Execute a chat completion with native clients and collect metrics.
    """
    client = get_llm_client(llm_profile)
    start_time = time.perf_counter()

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    cost = 0.0
    response_data = None
    response_code = 200
    error_message = None

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

        prompt_tokens = result["usage"]["prompt_tokens"]
        completion_tokens = result["usage"]["completion_tokens"]
        total_tokens = result["usage"]["total_tokens"]
        cost = result["cost"]
        response_data = result["raw_response"]

        return result["content"]

    except Exception as e:
        response_code = 500
        error_message = str(e)
        logger.error(f"Error in chat_completion_with_stats: {e}", exc_info=True)
        response_data = {
            "error": error_message,
            "error_type": type(e).__name__,
        }
        request_info["error"] = error_message
        request_info["error_type"] = type(e).__name__
        raise e
    finally:
        duration_ms = int((time.perf_counter() - start_time) * 1000)

        if session:
            try:
                stat_data = {
                    "llm_profile_id": llm_profile.id,
                    "name": llm_profile.name,
                    "response_code": response_code,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "duration_ms": duration_ms,
                    "cost": cost,
                    "currency": "USD",
                    "request_data": {"requests": [request_info]},
                    "response_data": response_data,
                }
                async with session.begin_nested():
                    await crud.insert(session, LLMCallStat, stat_data)
            except Exception as db_err:
                logger.error(f"Failed to save LLM call stats: {db_err}")
                # We don't want to rollback the whole session if saving stats fails,
                # but crud.insert might have already failed.
