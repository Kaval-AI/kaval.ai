import time
import math
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel
from google import genai
from google.genai import types
from kavalai.prices.gemini import GEMINI_PRICES


class GeminiClient:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        response_model: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()

        # Convert messages to Gemini format
        contents = self._convert_messages(messages)

        config_kwargs = {}
        if response_model:
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

        try:
            # Using async call
            response = await self.client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

            duration = time.perf_counter() - start_time

            content = response.text
            if response_model:
                try:
                    content = response_model.model_validate_json(content)
                except Exception:
                    # Fallback or re-raise? Let's try to be robust
                    pass

            usage_metadata = response.usage_metadata
            prompt_tokens = usage_metadata.prompt_token_count or 0
            completion_tokens = usage_metadata.candidates_token_count or 0
            total_tokens = usage_metadata.total_token_count or 0
            cached_tokens = usage_metadata.cached_content_token_count or 0

            cost = self.calculate_cost(
                model, prompt_tokens, completion_tokens, cached_tokens
            )

            return {
                "content": content,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cached_tokens": cached_tokens,
                },
                "duration": duration,
                "cost": cost,
                "raw_response": str(response),
                "model": model,
            }
        except Exception as e:
            raise e

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[types.Content]:
        gemini_messages = []
        for m in messages:
            role = m["role"]
            if role == "assistant":
                role = "model"
            elif role == "system":
                # The new SDK handles system instructions differently if we want them as system instructions
                # But for compatibility with existing messages, we'll keep them as they are or
                # we could pass them to system_instruction in config.
                # However, _convert_messages is used for 'contents' parameter.
                pass

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

        # Gemini uses 'type' but sometimes pydantic uses 'type' in a way that matches,
        # but let's ensure it's what Gemini expects if needed.

        if "properties" in schema:
            for prop in schema["properties"].values():
                self._cleanup_schema(prop)

        if "items" in schema:
            self._cleanup_schema(schema["items"])

    def calculate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int
    ) -> float:
        pricing = None
        # Try exact match first
        if model in GEMINI_PRICES:
            pricing = GEMINI_PRICES[model]
        else:
            # Try prefix match (e.g. gemini-1.5-pro-latest -> gemini-1.5-pro)
            for m, p in GEMINI_PRICES.items():
                if model.startswith(m):
                    pricing = p
                    break

        if not pricing:
            return 0.0

        return pricing.calculate_cost(prompt_tokens, completion_tokens, cached_tokens)

    async def compute_embeddings(
        self,
        model: str,
        texts: List[str],
        normalize: bool = False,
        **kwargs,
    ) -> List[List[float]]:
        response = await self.client.aio.models.embed_content(
            model=model,
            contents=texts,
            **kwargs,
        )
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
            return normalized_embeddings

        return embeddings
