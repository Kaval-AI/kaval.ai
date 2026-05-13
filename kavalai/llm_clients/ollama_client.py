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

import json
import os
import time
from typing import Any, Dict, List, Optional, Type, Tuple

from loguru import logger
import ollama
from pydantic import BaseModel

from kavalai.agents.db import ModelCallStat
from kavalai.llm_clients.common import (
    create_model_call_stat,
    get_model_name,
    safe_parse_json,
    safe_model_validate,
    Streamer,
)
from kavalai.normalizer import Normalizer, get_default_normalizer


class OllamaClient:
    def __init__(
        self,
        host: Optional[str] = None,
        timeout: float = 30.0,
    ):
        if not host:
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = timeout
        self.client = ollama.AsyncClient(host=host, timeout=self.timeout)

    async def chat_completions(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        response_model: Optional[Type[BaseModel]] = None,
        streamer: Optional[Streamer] = None,
        stream_delta: bool = False,
        **kwargs,
    ) -> Tuple[Any, ModelCallStat]:
        start_time = time.perf_counter()
        model_name = get_model_name(model)

        # Convert messages to Ollama format if needed (they are already in OpenAI-like format)
        # Ollama's chat expects [{'role': 'user', 'content': '...'}]

        call_kwargs = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            **{k: v for k, v in kwargs.items() if v is not None},
        }

        if response_model:
            # Ollama supports 'format': 'json' or a JSON schema (in newer versions)
            # For now, we'll use 'json' format and rely on our fix_json and pydantic validation
            call_kwargs["format"] = "json"

        full_content = ""
        prompt_tokens = 0
        completion_tokens = 0

        async for chunk in await self.client.chat(**call_kwargs):
            if "message" in chunk and "content" in chunk["message"]:
                delta = chunk["message"]["content"]
                full_content += delta
                if streamer is not None:
                    if stream_delta:
                        await streamer.stream_partial(delta)
                    else:
                        value = (
                            safe_parse_json(full_content)
                            if response_model
                            else full_content
                        )
                        if isinstance(value, (dict, list)):
                            value = json.dumps(value)
                        await streamer.stream_partial(value)

            if chunk.get("done"):
                prompt_tokens = chunk.get("prompt_eval_count", 0)
                completion_tokens = chunk.get("eval_count", 0)

        if streamer is not None:
            value = safe_parse_json(full_content) if response_model else full_content
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            await streamer.stream_complete(value)

        if response_model:
            result = safe_model_validate(response_model, full_content)
        else:
            result = full_content

        duration = time.perf_counter() - start_time

        stats = create_model_call_stat(
            call_type="llm",
            model=f"ollama/{model_name}",
            duration_sections=duration,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=None,
            response_data=result.model_dump()
            if hasattr(result, "model_dump")
            else result,
        )
        stats.currency = None
        return result, stats

    async def compute_embeddings(
        self,
        model: str,
        texts: List[str],
        normalize: bool = False,
        normalizer: Optional[Normalizer] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Tuple[List[List[float]], ModelCallStat]:
        start_time = time.perf_counter()
        model_name = get_model_name(model)

        # Ollama's embed API takes a single string or a list of strings
        # But it returns embeddings for them.

        embeddings = []
        total_prompt_tokens = 0

        # Ollama client doesn't have a batch embed that returns all at once in some versions,
        # but let's try the modern way
        for text in texts:
            response = await self.client.embed(model=model_name, input=text, **kwargs)
            embeddings.extend(response.get("embeddings", []))
            total_prompt_tokens += response.get("prompt_eval_count", 0)

        if normalize:
            if normalizer is None:
                normalizer = get_default_normalizer()
            embeddings = normalizer.transform(embeddings)

        duration = time.perf_counter() - start_time

        stats = create_model_call_stat(
            call_type="embedding",
            model=f"ollama/{model_name}",
            duration_sections=duration,
            batch_size=len(texts),
            total_tokens=total_prompt_tokens,
            cost=None,
            response_data=None,
        )
        stats.currency = None
        return embeddings, stats

    async def list_models(self) -> List[str]:
        try:
            response = await self.client.list()
            # Some versions of ollama-python return an object where 'models' is a list of model objects
            # Or response itself might be an object with 'models' attribute
            if hasattr(response, "models"):
                models = response.models
            elif isinstance(response, dict):
                models = response.get("models", [])
            else:
                models = []

            if not models:
                return []

            # Check if models are objects or dicts
            result = []
            for m in models:
                if hasattr(m, "model"):
                    result.append(m.model)
                elif isinstance(m, dict):
                    result.append(m.get("model", m.get("name")))
                else:
                    result.append(str(m))
            return result
        except Exception as e:
            logger.error(f"Ollama list_models error: {e}")
            return []
