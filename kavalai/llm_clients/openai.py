import time
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel
from openai import AsyncOpenAI
from kavalai.prices.openai import OPENAI_TEXT_PRICES


class OpenAIClient:
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        response_model: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()

        call_kwargs = {"model": model, "messages": messages, **kwargs}

        if response_model:
            call_kwargs["response_format"] = response_model

        try:
            if response_model:
                # Using OpenAI's native structured output (parse)
                response = await self.client.beta.chat.completions.parse(**call_kwargs)
                content = response.choices[0].message.parsed
            else:
                response = await self.client.chat.completions.create(**call_kwargs)
                content = response.choices[0].message.content

            duration = time.perf_counter() - start_time

            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0

            # OpenAI specific: cached tokens
            cached_tokens = 0
            if (
                usage
                and hasattr(usage, "prompt_tokens_details")
                and usage.prompt_tokens_details
            ):
                cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0)

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
                "raw_response": response.model_dump()
                if hasattr(response, "model_dump")
                else str(response),
                "model": model,
            }
        except Exception as e:
            # duration = time.perf_counter() - start_time
            raise e

    def calculate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int
    ) -> float:
        # Find matching model in price table
        pricing = None
        # Try exact match first
        if model in OPENAI_TEXT_PRICES:
            pricing = OPENAI_TEXT_PRICES[model]
        else:
            # Try prefix match
            for m, p in OPENAI_TEXT_PRICES.items():
                if model.startswith(m):
                    pricing = p
                    break

        if not pricing:
            return 0.0

        return pricing.calculate_cost(prompt_tokens, completion_tokens, cached_tokens)
