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

import json
from partial_json_parser import ensure_json
from pydantic import BaseModel

from kavalai.agents.db import ModelCallStat
from kavalai.agents.workflow_model import to_plain


def fix_json(data: str) -> Any:
    """Ensure string is valid JSON, handle leading/trailing characters and return dict/list."""
    # Find the start of JSON (first { or [)
    start_pos = -1
    for i, char in enumerate(data):
        if char in ("{", "["):
            start_pos = i
            break

    if start_pos == -1:
        # No JSON structure found, try to parse as is or return empty dict
        try:
            return json.loads(data)
        except Exception:
            return {}

    data = data[start_pos:]

    # Fast path: already valid JSON
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        # Handle extra trailing data
        if e.msg == "Extra data":
            try:
                return json.loads(data[: e.pos].strip())
            except Exception:
                pass

    # Try partial JSON parser
    try:
        fixed = ensure_json(data)
        return json.loads(fixed)
    except Exception:
        # Some partial_json_parser versions might fail on trailing commas.
        # Try a simple replacement for common case: ,} -> } and ,] -> ]
        try:
            fixed = ensure_json(data.replace(",}", "}").replace(",]", "]"))
            return json.loads(fixed)
        except Exception:
            pass

    # Fallback: find last valid closing brace/bracket
    for char in ("}", "]"):
        pos = data.rfind(char)
        if pos == -1:
            continue
        try:
            subset = data[: pos + 1]
            return json.loads(subset)
        except Exception:
            continue

    # Last resort: try to return whatever we have as dict/list if possible
    # or just an empty dict if all else fails
    return {}


class StreamContent(BaseModel):
    type: str
    name: str
    value: str


class Streamer:
    def __init__(self, name: str, queue: asyncio.Queue):
        self.name = name
        self.queue = queue

    async def stream_partial(self, value: str, name: Optional[str] = None):
        await self.queue.put(
            StreamContent(
                type="partial", name=name or self.name, value=value
            ).model_dump_json()
        )

    async def stream_complete(self, value: str, name: Optional[str] = None):
        await self.queue.put(
            StreamContent(
                type="complete", name=name or self.name, value=value
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
    duration_sections: float,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    batch_size: Optional[int] = None,
    response_code: int = 200,
    response_data: Any = None,
    cost: Optional[float] = None,
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
        duration_seconds=duration_sections,
        cost=cost,
        response_data=to_plain(response_data),
    )
