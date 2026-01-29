import time
import math
from typing import Any, Dict, List, Type, Tuple
from pydantic import BaseModel
from google import genai
from google.genai import types
from kavalai.agents.db import ModelCallStat


class GeminiClient:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        response_model: Type[BaseModel] = None,
        **kwargs,
    ) -> Tuple[Any, ModelCallStat]:
        start_time = time.perf_counter()

        # Convert messages to Gemini format
        contents = self._convert_messages(messages)

        config_kwargs = {}
        schema = response_model.model_json_schema()
        self._cleanup_schema(schema)
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = schema

        # Extract other kwargs for GenerateContentConfig
        for key in [
            "temperature",
            "top_p",
            "top_k",
            "max_output_tokens",
            "stop_sequences",
        ]:
            if key in kwargs:
                config_kwargs[key] = kwargs.pop(key)

        config = types.GenerateContentConfig(**config_kwargs)

        # Using async call
        response = await self.client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        duration = time.perf_counter() - start_time

        content = response_model.model_validate_json(response.text)

        usage_metadata = response.usage_metadata
        prompt_tokens = usage_metadata.prompt_token_count or 0
        completion_tokens = usage_metadata.candidates_token_count or 0
        total_tokens = usage_metadata.total_token_count or 0

        stats = ModelCallStat(
            call_type="llm",
            model=f"gemini/{model}",
            response_code=200,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            duration_seconds=duration,
            cost=None,  # We will compute cost later.
            response_data=str(response),
        )

        return content, stats

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[types.Content]:
        gemini_messages = []
        for m in messages:
            role = m["role"]
            if role == "assistant":
                role = "model"
            gemini_messages.append(
                types.Content(role=role, parts=[types.Part(text=m["content"])])
            )
        return gemini_messages

    def _cleanup_schema(self, schema: Dict[str, Any]):
        """Recursively remove 'title' and other unsupported fields from JSON schema for Gemini."""
        if not isinstance(schema, dict):
            return

        unsupported = ["title", "description", "default"]
        for field in unsupported:
            if field in schema:
                del schema[field]

        if "properties" in schema:
            for prop in schema["properties"].values():
                self._cleanup_schema(prop)

        if "items" in schema:
            self._cleanup_schema(schema["items"])

    async def compute_embeddings(
        self,
        model: str,
        texts: List[str],
        normalize: bool = False,
        **kwargs,
    ) -> Tuple[List[List[float]], ModelCallStat]:
        start_time = time.perf_counter()
        response = await self.client.aio.models.embed_content(
            model=model,
            contents=texts,
            **kwargs,
        )
        duration = time.perf_counter() - start_time

        # response.embeddings is a list of ContentEmbedding objects
        embeddings = [emb.values for emb in response.embeddings]

        if normalize:
            normalized_embeddings = []
            for emb in embeddings:
                norm = math.sqrt(sum(x * x for x in emb))
                if norm > 0:
                    normalized_embeddings.append([x / norm for x in emb])
                else:
                    normalized_embeddings.append(emb)
            embeddings = normalized_embeddings

        # Estimated tokens if not provided (Gemini embed_content doesn't always return usage in the same way as generate_content)
        # However, let's check if it exists
        total_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            total_tokens = response.usage_metadata.total_token_count or 0
        else:
            # Fallback: estimate tokens (rough estimation: 1 token ~= 4 chars)
            total_tokens = sum(len(t) for t in texts) // 4

        stats = ModelCallStat(
            call_type="embedding",
            model=f"gemini/{model}",
            response_code=200,
            batch_size=len(texts),
            total_tokens=total_tokens,
            duration_seconds=duration,
            response_data=str(response),
        )

        return embeddings, stats
