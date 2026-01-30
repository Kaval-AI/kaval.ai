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

from typing import Dict, Optional
from pydantic import BaseModel, Field


class TokenPricing(BaseModel):
    """Pricing for a specific token type (input, output, or cached)."""

    price_per_1m: float = 0.0
    # Optional tiered pricing, e.g., {"<=200k": 1.0, ">200k": 2.0}
    tiered: Optional[Dict[str, float]] = None

    def get_price(self, context_size: int = 0) -> float:
        if self.tiered:
            # Simple tiered logic for Gemini: <=200k or >200k
            if context_size > 200000 and ">200k" in self.tiered:
                return self.tiered[">200k"]
            return self.tiered.get("<=200k", self.price_per_1m)
        return self.price_per_1m


class ModelPricing(BaseModel):
    """Common structure for LLM model pricing."""

    model_name: str
    input: TokenPricing = Field(default_factory=TokenPricing)
    output: TokenPricing = Field(default_factory=TokenPricing)
    cached_input: Optional[TokenPricing] = None

    # Optional training/fine-tuning pricing
    training: Optional[float] = None

    def calculate_cost(
        self, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0
    ) -> float:
        context_size = prompt_tokens + completion_tokens

        input_p = self.input.get_price(context_size)
        output_p = self.output.get_price(context_size)

        cached_p = input_p  # Default to input price if not specified
        if self.cached_input:
            cached_p = self.cached_input.get_price(context_size)

        regular_input_tokens = prompt_tokens - cached_tokens

        cost = (
            (regular_input_tokens * input_p / 1_000_000)
            + (cached_tokens * cached_p / 1_000_000)
            + (completion_tokens * output_p / 1_000_000)
        )

        return cost
