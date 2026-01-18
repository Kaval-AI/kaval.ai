import logging
import os
from datetime import datetime, timezone
from uuid import UUID

import instructor
import time
import yaml
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from kavalai import crud
from kavalai.agents.db import LLMProfile, LLMCallStat

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


def get_instructor(llm_profile: LLMProfile) -> instructor.Instructor:
    logger.info(
        f"Creating LLM client for profile '{llm_profile.name}': {llm_profile.provider}/{llm_profile.model_name}"
    )
    params = dict(
        model=f"{llm_profile.provider}/{llm_profile.model_name}",
        async_client=True,
        mode=instructor.Mode.JSON,
    )
    if llm_profile.base_url:
        params["base_url"] = llm_profile.base_url
    if llm_profile.api_key:
        params["api_key"] = llm_profile.api_key
    if llm_profile.default_mode:
        params["mode"] = llm_profile.default_mode
    return instructor.from_provider(**params)


async def chat_completion_with_stats(
    llm_profile: LLMProfile,
    response_model: type[BaseModel],
    messages: list[dict],
    session: AsyncSession = None,
    **kwargs,
) -> any:
    """
    Execute a chat completion with instructor and collect metrics.
    """
    client = get_instructor(llm_profile)
    start_time = time.perf_counter()

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    response_data = None
    response_code = 200
    error_message = None
    all_requests = []

    def log_request(kwargs_in, exception=None):
        request_info = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "arguments": {k: v for k, v in kwargs_in.items() if k != "client"},
        }
        if exception:
            request_info["error"] = str(exception)
            request_info["error_type"] = type(exception).__name__
        all_requests.append(request_info)

    # To capture retries, we can wrap the create call or use instructor hooks if available.
    # Instructor uses tenacity. We can add a before_sleep callback to tenacity if we can access it.
    # However, instructor.from_provider() returns a wrapped client.

    # Another way is to wrap the messages/kwargs and use a proxy.
    # But simpler: use the 'max_retries' from kwargs if present, or default,
    # and manually loop if we want full control, OR just accept that instructor
    # might not easily expose individual attempts without more complex patching.

    # Actually, we can pass 'max_retries' to instructor.
    # Let's try to use a simple wrapper for the call to capture each attempt.

    async def wrapped_create():
        attempt = 0
        while True:
            attempt += 1
            try:
                # We don't want instructor to retry internally if we want to capture each attempt
                # but instructor's 'max_retries' is useful.
                # If we let instructor retry, we only get the last one.
                # To capture all, we can set max_retries=0 and handle retries here,
                # but that might break some instructor-specific logic.

                # BETTER: Instructor allows passing a custom tenacity Retrying object or config.
                # But it's not well documented for all providers.

                # Let's stick to capturing what we can.
                # If the user really wants ALL http requests, we might need to intercept at the httpx level.

                res = await client.chat.completions.create(
                    response_model=response_model,
                    messages=messages,
                    **kwargs,
                )
                log_request(
                    {
                        "messages": messages,
                        "response_model": str(response_model),
                        **kwargs,
                    }
                )
                return res
            except Exception as e:
                log_request(
                    {
                        "messages": messages,
                        "response_model": str(response_model),
                        **kwargs,
                    },
                    exception=e,
                )
                raise e

    try:
        response = await wrapped_create()

        # Extract usage if available (depends on provider)
        if hasattr(response, "_raw_response"):
            raw = response._raw_response
            if hasattr(raw, "usage") and raw.usage:
                prompt_tokens = getattr(raw.usage, "prompt_tokens", 0)
                completion_tokens = getattr(raw.usage, "completion_tokens", 0)
                total_tokens = getattr(raw.usage, "total_tokens", 0)

            # Capture response data
            if hasattr(raw, "model_dump"):
                response_data = raw.model_dump()
            elif hasattr(raw, "dict"):
                response_data = raw.dict()
            else:
                response_data = str(raw)

        log_request(
            {"messages": messages, "response_model": str(response_model), **kwargs}
        )
        return response

    except Exception as e:
        response_code = 500
        error_message = str(e)
        logger.error(f"Error in chat_completion_with_stats: {e}", exc_info=True)
        response_data = {
            "error": error_message,
            "error_type": type(e).__name__,
        }
        log_request(
            {"messages": messages, "response_model": str(response_model), **kwargs},
            exception=e,
        )
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
                    "request_data": {"requests": all_requests},
                    "response_data": response_data,
                }
                # Create a subtransaction or use a separate session to save stats
                # to avoid rolling back the main transaction if stats saving fails
                # or if the main transaction already failed.
                async with session.begin_nested():
                    await crud.insert(session, LLMCallStat, stat_data)
            except Exception as db_err:
                logger.error(f"Failed to save LLM call stats: {db_err}")
                # We don't want to rollback the whole session if saving stats fails,
                # but crud.insert might have already failed.
