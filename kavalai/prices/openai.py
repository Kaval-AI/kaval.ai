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

# Prices per 1M tokens in USD
# Text tokens
OPENAI_TEXT_PRICES: Dict[str, ModelPricing] = {
    "gpt-5.2": ModelPricing(
        model_name="gpt-5.2",
        input=TokenPricing(price_per_1m=1.75),
        cached_input=TokenPricing(price_per_1m=0.175),
        output=TokenPricing(price_per_1m=14.00),
    ),
    "gpt-5.1": ModelPricing(
        model_name="gpt-5.1",
        input=TokenPricing(price_per_1m=1.25),
        cached_input=TokenPricing(price_per_1m=0.125),
        output=TokenPricing(price_per_1m=10.00),
    ),
    "gpt-5": ModelPricing(
        model_name="gpt-5",
        input=TokenPricing(price_per_1m=1.25),
        cached_input=TokenPricing(price_per_1m=0.125),
        output=TokenPricing(price_per_1m=10.00),
    ),
    "gpt-5-mini": ModelPricing(
        model_name="gpt-5-mini",
        input=TokenPricing(price_per_1m=0.25),
        cached_input=TokenPricing(price_per_1m=0.025),
        output=TokenPricing(price_per_1m=2.00),
    ),
    "gpt-5-nano": ModelPricing(
        model_name="gpt-5-nano",
        input=TokenPricing(price_per_1m=0.05),
        cached_input=TokenPricing(price_per_1m=0.005),
        output=TokenPricing(price_per_1m=0.40),
    ),
    "gpt-5.2-chat-latest": ModelPricing(
        model_name="gpt-5.2-chat-latest",
        input=TokenPricing(price_per_1m=1.75),
        cached_input=TokenPricing(price_per_1m=0.175),
        output=TokenPricing(price_per_1m=14.00),
    ),
    "gpt-5.1-chat-latest": ModelPricing(
        model_name="gpt-5.1-chat-latest",
        input=TokenPricing(price_per_1m=1.25),
        cached_input=TokenPricing(price_per_1m=0.125),
        output=TokenPricing(price_per_1m=10.00),
    ),
    "gpt-5-chat-latest": ModelPricing(
        model_name="gpt-5-chat-latest",
        input=TokenPricing(price_per_1m=1.25),
        cached_input=TokenPricing(price_per_1m=0.125),
        output=TokenPricing(price_per_1m=10.00),
    ),
    "gpt-5.2-codex": ModelPricing(
        model_name="gpt-5.2-codex",
        input=TokenPricing(price_per_1m=1.75),
        cached_input=TokenPricing(price_per_1m=0.175),
        output=TokenPricing(price_per_1m=14.00),
    ),
    "gpt-5.1-codex-max": ModelPricing(
        model_name="gpt-5.1-codex-max",
        input=TokenPricing(price_per_1m=1.25),
        cached_input=TokenPricing(price_per_1m=0.125),
        output=TokenPricing(price_per_1m=10.00),
    ),
    "gpt-5.1-codex": ModelPricing(
        model_name="gpt-5.1-codex",
        input=TokenPricing(price_per_1m=1.25),
        cached_input=TokenPricing(price_per_1m=0.125),
        output=TokenPricing(price_per_1m=10.00),
    ),
    "gpt-5-codex": ModelPricing(
        model_name="gpt-5-codex",
        input=TokenPricing(price_per_1m=1.25),
        cached_input=TokenPricing(price_per_1m=0.125),
        output=TokenPricing(price_per_1m=10.00),
    ),
    "gpt-5.2-pro": ModelPricing(
        model_name="gpt-5.2-pro",
        input=TokenPricing(price_per_1m=21.00),
        output=TokenPricing(price_per_1m=168.00),
    ),
    "gpt-5-pro": ModelPricing(
        model_name="gpt-5-pro",
        input=TokenPricing(price_per_1m=15.00),
        output=TokenPricing(price_per_1m=120.00),
    ),
    "gpt-4.1": ModelPricing(
        model_name="gpt-4.1",
        input=TokenPricing(price_per_1m=2.00),
        cached_input=TokenPricing(price_per_1m=0.50),
        output=TokenPricing(price_per_1m=8.00),
    ),
    "gpt-4.1-mini": ModelPricing(
        model_name="gpt-4.1-mini",
        input=TokenPricing(price_per_1m=0.40),
        cached_input=TokenPricing(price_per_1m=0.10),
        output=TokenPricing(price_per_1m=1.60),
    ),
    "gpt-4.1-nano": ModelPricing(
        model_name="gpt-4.1-nano",
        input=TokenPricing(price_per_1m=0.10),
        cached_input=TokenPricing(price_per_1m=0.025),
        output=TokenPricing(price_per_1m=0.40),
    ),
    "gpt-4o": ModelPricing(
        model_name="gpt-4o",
        input=TokenPricing(price_per_1m=2.50),
        cached_input=TokenPricing(price_per_1m=1.25),
        output=TokenPricing(price_per_1m=10.00),
    ),
    "gpt-4o-2024-05-13": ModelPricing(
        model_name="gpt-4o-2024-05-13",
        input=TokenPricing(price_per_1m=5.00),
        output=TokenPricing(price_per_1m=15.00),
    ),
    "gpt-4o-mini": ModelPricing(
        model_name="gpt-4o-mini",
        input=TokenPricing(price_per_1m=0.15),
        cached_input=TokenPricing(price_per_1m=0.075),
        output=TokenPricing(price_per_1m=0.60),
    ),
    "gpt-realtime": ModelPricing(
        model_name="gpt-realtime",
        input=TokenPricing(price_per_1m=4.00),
        cached_input=TokenPricing(price_per_1m=0.40),
        output=TokenPricing(price_per_1m=16.00),
    ),
    "gpt-realtime-mini": ModelPricing(
        model_name="gpt-realtime-mini",
        input=TokenPricing(price_per_1m=0.60),
        cached_input=TokenPricing(price_per_1m=0.06),
        output=TokenPricing(price_per_1m=2.40),
    ),
    "gpt-4o-realtime-preview": ModelPricing(
        model_name="gpt-4o-realtime-preview",
        input=TokenPricing(price_per_1m=5.00),
        cached_input=TokenPricing(price_per_1m=2.50),
        output=TokenPricing(price_per_1m=20.00),
    ),
    "gpt-4o-mini-realtime-preview": ModelPricing(
        model_name="gpt-4o-mini-realtime-preview",
        input=TokenPricing(price_per_1m=0.60),
        cached_input=TokenPricing(price_per_1m=0.30),
        output=TokenPricing(price_per_1m=2.40),
    ),
    "gpt-audio": ModelPricing(
        model_name="gpt-audio",
        input=TokenPricing(price_per_1m=2.50),
        output=TokenPricing(price_per_1m=10.00),
    ),
    "gpt-audio-mini": ModelPricing(
        model_name="gpt-audio-mini",
        input=TokenPricing(price_per_1m=0.60),
        output=TokenPricing(price_per_1m=2.40),
    ),
    "gpt-4o-audio-preview": ModelPricing(
        model_name="gpt-4o-audio-preview",
        input=TokenPricing(price_per_1m=2.50),
        output=TokenPricing(price_per_1m=10.00),
    ),
    "gpt-4o-mini-audio-preview": ModelPricing(
        model_name="gpt-4o-mini-audio-preview",
        input=TokenPricing(price_per_1m=0.15),
        output=TokenPricing(price_per_1m=0.60),
    ),
    "o1": ModelPricing(
        model_name="o1",
        input=TokenPricing(price_per_1m=15.00),
        cached_input=TokenPricing(price_per_1m=7.50),
        output=TokenPricing(price_per_1m=60.00),
    ),
    "o1-pro": ModelPricing(
        model_name="o1-pro",
        input=TokenPricing(price_per_1m=150.00),
        output=TokenPricing(price_per_1m=600.00),
    ),
    "o3-pro": ModelPricing(
        model_name="o3-pro",
        input=TokenPricing(price_per_1m=20.00),
        output=TokenPricing(price_per_1m=80.00),
    ),
    "o3": ModelPricing(
        model_name="o3",
        input=TokenPricing(price_per_1m=2.00),
        cached_input=TokenPricing(price_per_1m=0.50),
        output=TokenPricing(price_per_1m=8.00),
    ),
    "o3-deep-research": ModelPricing(
        model_name="o3-deep-research",
        input=TokenPricing(price_per_1m=10.00),
        cached_input=TokenPricing(price_per_1m=2.50),
        output=TokenPricing(price_per_1m=40.00),
    ),
    "o4-mini": ModelPricing(
        model_name="o4-mini",
        input=TokenPricing(price_per_1m=1.10),
        cached_input=TokenPricing(price_per_1m=0.275),
        output=TokenPricing(price_per_1m=4.40),
    ),
    "o4-mini-deep-research": ModelPricing(
        model_name="o4-mini-deep-research",
        input=TokenPricing(price_per_1m=2.00),
        cached_input=TokenPricing(price_per_1m=0.50),
        output=TokenPricing(price_per_1m=8.00),
    ),
    "o3-mini": ModelPricing(
        model_name="o3-mini",
        input=TokenPricing(price_per_1m=1.10),
        cached_input=TokenPricing(price_per_1m=0.55),
        output=TokenPricing(price_per_1m=4.40),
    ),
    "o1-mini": ModelPricing(
        model_name="o1-mini",
        input=TokenPricing(price_per_1m=1.10),
        cached_input=TokenPricing(price_per_1m=0.55),
        output=TokenPricing(price_per_1m=4.40),
    ),
    "gpt-5.1-codex-mini": ModelPricing(
        model_name="gpt-5.1-codex-mini",
        input=TokenPricing(price_per_1m=0.25),
        cached_input=TokenPricing(price_per_1m=0.025),
        output=TokenPricing(price_per_1m=2.00),
    ),
    "codex-mini-latest": ModelPricing(
        model_name="codex-mini-latest",
        input=TokenPricing(price_per_1m=1.50),
        cached_input=TokenPricing(price_per_1m=0.375),
        output=TokenPricing(price_per_1m=6.00),
    ),
    "gpt-5-search-api": ModelPricing(
        model_name="gpt-5-search-api",
        input=TokenPricing(price_per_1m=1.25),
        cached_input=TokenPricing(price_per_1m=0.125),
        output=TokenPricing(price_per_1m=10.00),
    ),
    "gpt-4o-mini-search-preview": ModelPricing(
        model_name="gpt-4o-mini-search-preview",
        input=TokenPricing(price_per_1m=0.15),
        output=TokenPricing(price_per_1m=0.60),
    ),
    "gpt-4o-search-preview": ModelPricing(
        model_name="gpt-4o-search-preview",
        input=TokenPricing(price_per_1m=2.50),
        output=TokenPricing(price_per_1m=10.00),
    ),
    "computer-use-preview": ModelPricing(
        model_name="computer-use-preview",
        input=TokenPricing(price_per_1m=3.00),
        output=TokenPricing(price_per_1m=12.00),
    ),
    "gpt-image-1.5": ModelPricing(
        model_name="gpt-image-1.5",
        input=TokenPricing(price_per_1m=5.00),
        cached_input=TokenPricing(price_per_1m=1.25),
        output=TokenPricing(price_per_1m=10.00),
    ),
    "chatgpt-image-latest": ModelPricing(
        model_name="chatgpt-image-latest",
        input=TokenPricing(price_per_1m=5.00),
        cached_input=TokenPricing(price_per_1m=1.25),
        output=TokenPricing(price_per_1m=10.00),
    ),
    "gpt-image-1": ModelPricing(
        model_name="gpt-image-1",
        input=TokenPricing(price_per_1m=5.00),
        cached_input=TokenPricing(price_per_1m=1.25),
    ),
    "gpt-image-1-mini": ModelPricing(
        model_name="gpt-image-1-mini",
        input=TokenPricing(price_per_1m=2.00),
        cached_input=TokenPricing(price_per_1m=0.20),
    ),
}

# Audio tokens
# Prices per 1M tokens
OPENAI_AUDIO_PRICES: Dict[str, Dict[str, Any]] = {
    "gpt-realtime": {"input": 32.00, "cached_input": 0.40, "output": 64.00},
    "gpt-realtime-mini": {"input": 10.00, "cached_input": 0.30, "output": 20.00},
    "gpt-4o-realtime-preview": {"input": 40.00, "cached_input": 2.50, "output": 80.00},
    "gpt-4o-mini-realtime-preview": {
        "input": 10.00,
        "cached_input": 0.30,
        "output": 20.00,
    },
    "gpt-audio": {"input": 32.00, "cached_input": None, "output": 64.00},
    "gpt-audio-mini": {"input": 10.00, "cached_input": None, "output": 20.00},
    "gpt-4o-audio-preview": {"input": 40.00, "cached_input": None, "output": 80.00},
    "gpt-4o-mini-audio-preview": {
        "input": 10.00,
        "cached_input": None,
        "output": 20.00,
    },
}

# Fine-tuning
# Prices per 1M tokens
OPENAI_FINE_TUNING_PRICES: Dict[str, ModelPricing] = {
    "o4-mini-2025-04-16": ModelPricing(
        model_name="o4-mini-2025-04-16",
        training=100.00,
        input=TokenPricing(price_per_1m=4.00),
        cached_input=TokenPricing(price_per_1m=1.00),
        output=TokenPricing(price_per_1m=16.00),
    ),
    "o4-mini-2025-04-16-data-sharing": ModelPricing(
        model_name="o4-mini-2025-04-16-data-sharing",
        training=100.00,
        input=TokenPricing(price_per_1m=2.00),
        cached_input=TokenPricing(price_per_1m=0.50),
        output=TokenPricing(price_per_1m=8.00),
    ),
    "gpt-4.1-2025-04-14": ModelPricing(
        model_name="gpt-4.1-2025-04-14",
        training=25.00,
        input=TokenPricing(price_per_1m=3.00),
        cached_input=TokenPricing(price_per_1m=0.75),
        output=TokenPricing(price_per_1m=12.00),
    ),
    "gpt-4.1-mini-2025-04-14": ModelPricing(
        model_name="gpt-4.1-mini-2025-04-14",
        training=5.00,
        input=TokenPricing(price_per_1m=0.80),
        cached_input=TokenPricing(price_per_1m=0.20),
        output=TokenPricing(price_per_1m=3.20),
    ),
    "gpt-4.1-nano-2025-04-14": ModelPricing(
        model_name="gpt-4.1-nano-2025-04-14",
        training=1.50,
        input=TokenPricing(price_per_1m=0.20),
        cached_input=TokenPricing(price_per_1m=0.05),
        output=TokenPricing(price_per_1m=0.80),
    ),
    "gpt-4o-2024-08-06": ModelPricing(
        model_name="gpt-4o-2024-08-06",
        training=25.00,
        input=TokenPricing(price_per_1m=3.75),
        cached_input=TokenPricing(price_per_1m=1.875),
        output=TokenPricing(price_per_1m=15.00),
    ),
    "gpt-4o-mini-2024-07-18": ModelPricing(
        model_name="gpt-4o-mini-2024-07-18",
        training=3.00,
        input=TokenPricing(price_per_1m=0.30),
        cached_input=TokenPricing(price_per_1m=0.15),
        output=TokenPricing(price_per_1m=1.20),
    ),
    "gpt-3.5-turbo": ModelPricing(
        model_name="gpt-3.5-turbo",
        training=8.00,
        input=TokenPricing(price_per_1m=3.00),
        output=TokenPricing(price_per_1m=6.00),
    ),
    "davinci-002": ModelPricing(
        model_name="davinci-002",
        training=6.00,
        input=TokenPricing(price_per_1m=12.00),
        output=TokenPricing(price_per_1m=12.00),
    ),
    "babbage-002": ModelPricing(
        model_name="babbage-002",
        training=0.40,
        input=TokenPricing(price_per_1m=1.60),
        output=TokenPricing(price_per_1m=1.60),
    ),
}

# Built-in tools
OPENAI_TOOL_PRICES: Dict[str, Any] = {
    "code_interpreter": {
        "1 GB": 0.03,
        "4 GB": 0.12,
        "16 GB": 0.48,
        "64 GB": 1.92,
    },
    "file_search_storage": 0.10,  # per GB per day (1GB free)
    "file_search_tool_call": 2.50,  # per 1k calls (Responses API only)
    "web_search": 10.00,  # per 1k calls + tokens
    "web_search_preview_reasoning": 10.00,  # per 1k calls + tokens
    "web_search_preview_non_reasoning": 25.00,  # per 1k calls + tokens
}

# AgentKit
OPENAI_AGENTKIT_PRICES: Dict[str, Any] = {
    "chatkit_storage": 0.10,  # per GB-day (1GB free)
}

# Transcription and speech generation
# Prices per 1M tokens
OPENAI_TRANSCRIPTION_SPEECH_TEXT_PRICES: Dict[str, Dict[str, Any]] = {
    "gpt-4o-mini-tts": {
        "input": 0.60,
        "output": None,
        "estimated_cost_per_minute": 0.015,
    },
    "gpt-4o-transcribe": {
        "input": 2.50,
        "output": 10.00,
        "estimated_cost_per_minute": 0.006,
    },
    "gpt-4o-transcribe-diarize": {
        "input": 2.50,
        "output": 10.00,
        "estimated_cost_per_minute": 0.006,
    },
    "gpt-4o-mini-transcribe": {
        "input": 1.25,
        "output": 5.00,
        "estimated_cost_per_minute": 0.003,
    },
}

OPENAI_TRANSCRIPTION_SPEECH_AUDIO_PRICES: Dict[str, Dict[str, Any]] = {
    "gpt-4o-mini-tts": {
        "input": None,
        "output": 12.00,
        "estimated_cost_per_minute": 0.015,
    },
    "gpt-4o-transcribe": {
        "input": 6.00,
        "output": None,
        "estimated_cost_per_minute": 0.006,
    },
    "gpt-4o-transcribe-diarize": {
        "input": 6.00,
        "output": None,
        "estimated_cost_per_minute": 0.006,
    },
    "gpt-4o-mini-transcribe": {
        "input": 3.00,
        "output": None,
        "estimated_cost_per_minute": 0.003,
    },
}

OPENAI_TRANSCRIPTION_SPEECH_OTHER_PRICES: Dict[str, Any] = {
    "whisper": 0.006,  # per minute
    "tts": 15.00,  # per 1M characters
    "tts_hd": 30.00,  # per 1M characters
}

# Embeddings
# Prices per 1M tokens
OPENAI_EMBEDDING_PRICES: Dict[str, Dict[str, float]] = {
    "text-embedding-3-small": {"standard": 0.02, "batch": 0.01},
    "text-embedding-3-large": {"standard": 0.13, "batch": 0.065},
    "text-embedding-ada-002": {"standard": 0.10, "batch": 0.05},
}

# Legacy models
# Prices per 1M tokens
OPENAI_LEGACY_PRICES: Dict[str, ModelPricing] = {
    "chatgpt-4o-latest": ModelPricing(
        model_name="chatgpt-4o-latest",
        input=TokenPricing(price_per_1m=5.00),
        output=TokenPricing(price_per_1m=15.00),
    ),
    "gpt-4-turbo-2024-04-09": ModelPricing(
        model_name="gpt-4-turbo-2024-04-09",
        input=TokenPricing(price_per_1m=10.00),
        output=TokenPricing(price_per_1m=30.00),
    ),
    "gpt-4-0125-preview": ModelPricing(
        model_name="gpt-4-0125-preview",
        input=TokenPricing(price_per_1m=10.00),
        output=TokenPricing(price_per_1m=30.00),
    ),
    "gpt-4-1106-preview": ModelPricing(
        model_name="gpt-4-1106-preview",
        input=TokenPricing(price_per_1m=10.00),
        output=TokenPricing(price_per_1m=30.00),
    ),
    "gpt-4-1106-vision-preview": ModelPricing(
        model_name="gpt-4-1106-vision-preview",
        input=TokenPricing(price_per_1m=10.00),
        output=TokenPricing(price_per_1m=30.00),
    ),
    "gpt-4-0613": ModelPricing(
        model_name="gpt-4-0613",
        input=TokenPricing(price_per_1m=30.00),
        output=TokenPricing(price_per_1m=60.00),
    ),
    "gpt-4-0314": ModelPricing(
        model_name="gpt-4-0314",
        input=TokenPricing(price_per_1m=30.00),
        output=TokenPricing(price_per_1m=60.00),
    ),
    "gpt-4-32k": ModelPricing(
        model_name="gpt-4-32k",
        input=TokenPricing(price_per_1m=60.00),
        output=TokenPricing(price_per_1m=120.00),
    ),
    "gpt-3.5-turbo": ModelPricing(
        model_name="gpt-3.5-turbo",
        input=TokenPricing(price_per_1m=0.50),
        output=TokenPricing(price_per_1m=1.50),
    ),
    "gpt-3.5-turbo-0125": ModelPricing(
        model_name="gpt-3.5-turbo-0125",
        input=TokenPricing(price_per_1m=0.50),
        output=TokenPricing(price_per_1m=1.50),
    ),
    "gpt-3.5-turbo-1106": ModelPricing(
        model_name="gpt-3.5-turbo-1106",
        input=TokenPricing(price_per_1m=1.00),
        output=TokenPricing(price_per_1m=2.00),
    ),
    "gpt-3.5-turbo-0613": ModelPricing(
        model_name="gpt-3.5-turbo-0613",
        input=TokenPricing(price_per_1m=1.50),
        output=TokenPricing(price_per_1m=2.00),
    ),
    "gpt-3.5-0301": ModelPricing(
        model_name="gpt-3.5-0301",
        input=TokenPricing(price_per_1m=1.50),
        output=TokenPricing(price_per_1m=2.00),
    ),
    "gpt-3.5-turbo-instruct": ModelPricing(
        model_name="gpt-3.5-turbo-instruct",
        input=TokenPricing(price_per_1m=1.50),
        output=TokenPricing(price_per_1m=2.00),
    ),
    "gpt-3.5-turbo-16k-0613": ModelPricing(
        model_name="gpt-3.5-turbo-16k-0613",
        input=TokenPricing(price_per_1m=3.00),
        output=TokenPricing(price_per_1m=4.00),
    ),
    "davinci-002": ModelPricing(
        model_name="davinci-002",
        input=TokenPricing(price_per_1m=2.00),
        output=TokenPricing(price_per_1m=2.00),
    ),
    "babbage-002": ModelPricing(
        model_name="babbage-002",
        input=TokenPricing(price_per_1m=0.40),
        output=TokenPricing(price_per_1m=0.40),
    ),
}


def get_openai_chat_cost(
    model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0
) -> float:
    """
    Compute the cost of an OpenAI chat completion call in USD.
    """
    # Try text prices first
    pricing = OPENAI_TEXT_PRICES.get(model)
    if not pricing:
        # Try legacy prices
        pricing = OPENAI_LEGACY_PRICES.get(model)

    if not pricing:
        # Fallback to some default or 0 if unknown
        return 0.0

    return pricing.calculate_cost(prompt_tokens, completion_tokens, cached_tokens)


def get_openai_embedding_cost(model: str, tokens: int, mode: str = "standard") -> float:
    """
    Compute the cost of an OpenAI embedding call in USD.
    """
    model_prices = OPENAI_EMBEDDING_PRICES.get(model)
    if not model_prices:
        return 0.0

    price_per_1m = model_prices.get(mode, 0.0)
    return tokens * price_per_1m / 1_000_000
