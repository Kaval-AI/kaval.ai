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

import asyncio
import random
from typing import Any, Callable, TypeVar

import openai
from google.genai import errors
from loguru import logger

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
