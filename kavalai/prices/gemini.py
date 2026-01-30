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

from typing import Dict, Any
from kavalai.prices.common import ModelPricing, TokenPricing

# Prices per 1M tokens in USD unless otherwise specified
GEMINI_PRICES: Dict[str, ModelPricing] = {
    "gemini-3-pro-preview": ModelPricing(
        model_name="gemini-3-pro-preview",
        input=TokenPricing(tiered={"<=200k": 1.00, ">200k": 2.00}),
        output=TokenPricing(tiered={"<=200k": 6.00, ">200k": 9.00}),
        cached_input=TokenPricing(tiered={"<=200k": 0.20, ">200k": 0.40}),
    ),
    "gemini-3-flash-preview": ModelPricing(
        model_name="gemini-3-flash-preview",
        input=TokenPricing(price_per_1m=0.25),
        output=TokenPricing(price_per_1m=1.50),
        cached_input=TokenPricing(price_per_1m=0.05),
    ),
    "gemini-3-pro-image-preview": ModelPricing(
        model_name="gemini-3-pro-image-preview",
        input=TokenPricing(price_per_1m=1.00),
        output=TokenPricing(price_per_1m=6.00),
    ),
    "gemini-2.5-pro": ModelPricing(
        model_name="gemini-2.5-pro",
        input=TokenPricing(tiered={"<=200k": 0.625, ">200k": 1.25}),
        output=TokenPricing(tiered={"<=200k": 5.00, ">200k": 7.50}),
        cached_input=TokenPricing(tiered={"<=200k": 0.125, ">200k": 0.25}),
    ),
    "gemini-2.5-flash": ModelPricing(
        model_name="gemini-2.5-flash",
        input=TokenPricing(price_per_1m=0.15),
        output=TokenPricing(price_per_1m=1.25),
        cached_input=TokenPricing(price_per_1m=0.03),
    ),
    "gemini-2.5-flash-preview-09-2025": ModelPricing(
        model_name="gemini-2.5-flash-preview-09-2025",
        input=TokenPricing(price_per_1m=0.15),
        output=TokenPricing(price_per_1m=1.25),
        cached_input=TokenPricing(price_per_1m=0.03),
    ),
    "gemini-2.5-flash-lite": ModelPricing(
        model_name="gemini-2.5-flash-lite",
        input=TokenPricing(price_per_1m=0.05),
        output=TokenPricing(price_per_1m=0.20),
        cached_input=TokenPricing(price_per_1m=0.01),
    ),
    "gemini-2.5-flash-lite-preview-09-2025": ModelPricing(
        model_name="gemini-2.5-flash-lite-preview-09-2025",
        input=TokenPricing(price_per_1m=0.05),
        output=TokenPricing(price_per_1m=0.20),
        cached_input=TokenPricing(price_per_1m=0.01),
    ),
    "gemini-2.5-flash-native-audio-preview-12-2025": ModelPricing(
        model_name="gemini-2.5-flash-native-audio-preview-12-2025",
        input=TokenPricing(price_per_1m=0.50),
        output=TokenPricing(price_per_1m=2.00),
    ),
    "gemini-2.5-flash-image": ModelPricing(
        model_name="gemini-2.5-flash-image",
        input=TokenPricing(price_per_1m=0.15),
    ),
    "gemini-2.5-flash-preview-tts": ModelPricing(
        model_name="gemini-2.5-flash-preview-tts",
        input=TokenPricing(price_per_1m=0.25),
    ),
    "gemini-2.5-pro-preview-tts": ModelPricing(
        model_name="gemini-2.5-pro-preview-tts",
        input=TokenPricing(price_per_1m=0.50),
    ),
    "gemini-2.0-flash": ModelPricing(
        model_name="gemini-2.0-flash",
        input=TokenPricing(price_per_1m=0.05),
        output=TokenPricing(price_per_1m=0.20),
        cached_input=TokenPricing(price_per_1m=0.025),
    ),
    "gemini-2.0-flash-lite": ModelPricing(
        model_name="gemini-2.0-flash-lite",
        input=TokenPricing(price_per_1m=0.0375),
        output=TokenPricing(price_per_1m=0.15),
    ),
    "gemini-embedding-001": ModelPricing(
        model_name="gemini-embedding-001",
        input=TokenPricing(price_per_1m=0.075),
    ),
    "gemini-2.5-computer-use-preview-10-2025": ModelPricing(
        model_name="gemini-2.5-computer-use-preview-10-2025",
        input=TokenPricing(tiered={"<=200k": 1.25, ">200k": 2.50}),
        output=TokenPricing(tiered={"<=200k": 10.00, ">200k": 15.00}),
    ),
}

# Image Generation - Prices per image in USD
GEMINI_IMAGE_GENERATION_PRICES: Dict[str, Dict[str, float]] = {
    "imagen-4.0-fast-generate-001": 0.02,
    "imagen-4.0-generate-001": 0.04,
    "imagen-4.0-ultra-generate-001": 0.06,
    "imagen-3.0-generate-002": 0.03,
}

# Video Generation - Prices per second in USD
GEMINI_VIDEO_GENERATION_PRICES: Dict[str, Dict[str, Any]] = {
    "veo-3.1-generate-preview": {
        "720p_1080p": 0.40,
        "4k": 0.60,
    },
    "veo-3.1-fast-generate-preview": {
        "720p_1080p": 0.15,
        "4k": 0.35,
    },
    "veo-3.0-generate-001": 0.40,
    "veo-3.0-fast-generate-001": 0.15,
    "veo-2.0-generate-001": 0.35,
}
