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
from typing import Any, Optional

from pydantic import BaseModel

from kavalai.agents.db import ModelCallStat


class StreamContent(BaseModel):
    type: str
    name: str
    value: str


class Streamer:
    def __init__(self, name: str, queue: asyncio.Queue):
        self.name = name
        self.queue = queue

    async def stream_partial(self, value: str):
        await self.queue.put(
            StreamContent(type="partial", name=self.name, value=value).model_dump_json()
        )

    async def stream_complete(self, value: str):
        await self.queue.put(
            StreamContent(
                type="complete", name=self.name, value=value
            ).model_dump_json()
        )


def get_model_name(model: str) -> str:
    """Extract model name from 'provider/model' syntax."""
    if "/" in model:
        return model.split("/")[-1]
    return model


def create_model_call_stat(
    call_type: str,
    model: str,
    duration: float,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    batch_size: Optional[int] = None,
    response_code: int = 200,
    response_data: Any = None,
) -> ModelCallStat:
    if (
        total_tokens is None
        and prompt_tokens is not None
        and completion_tokens is not None
    ):
        total_tokens = prompt_tokens + completion_tokens

    return ModelCallStat(
        call_type=call_type,
        model=model,
        response_code=response_code,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        batch_size=batch_size,
        duration_seconds=duration,
        cost=None,  # We will compute cost later
        response_data=response_data
        if isinstance(response_data, (dict, list)) or response_data is None
        else str(response_data),
    )
