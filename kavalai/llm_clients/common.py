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

import math
from typing import Any, List, Optional
from pydantic import BaseModel
from kavalai.agents.db import ModelCallStat


class StreamContent(BaseModel):
    type: str
    name: str
    value: str


def normalize_embeddings(embeddings: List[List[float]]) -> List[List[float]]:
    normalized_embeddings = []
    for emb in embeddings:
        norm = math.sqrt(sum(x * x for x in emb))
        if norm > 0:
            normalized_embeddings.append([x / norm for x in emb])
        else:
            normalized_embeddings.append(emb)
    return normalized_embeddings


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
