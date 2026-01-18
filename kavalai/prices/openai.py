from typing import Dict, Any

# Prices per 1M tokens in USD
# Text tokens
OPENAI_TEXT_PRICES: Dict[str, Dict[str, Any]] = {
    "gpt-5.2": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "gpt-5.1": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
    "gpt-5.2-chat-latest": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "gpt-5.1-chat-latest": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-chat-latest": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5.2-codex": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "gpt-5.1-codex-max": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5.1-codex": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-codex": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5.2-pro": {"input": 21.00, "cached_input": None, "output": 168.00},
    "gpt-5-pro": {"input": 15.00, "cached_input": None, "output": 120.00},
    "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-2024-05-13": {"input": 5.00, "cached_input": None, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "gpt-realtime": {"input": 4.00, "cached_input": 0.40, "output": 16.00},
    "gpt-realtime-mini": {"input": 0.60, "cached_input": 0.06, "output": 2.40},
    "gpt-4o-realtime-preview": {"input": 5.00, "cached_input": 2.50, "output": 20.00},
    "gpt-4o-mini-realtime-preview": {
        "input": 0.60,
        "cached_input": 0.30,
        "output": 2.40,
    },
    "gpt-audio": {"input": 2.50, "cached_input": None, "output": 10.00},
    "gpt-audio-mini": {"input": 0.60, "cached_input": None, "output": 2.40},
    "gpt-4o-audio-preview": {"input": 2.50, "cached_input": None, "output": 10.00},
    "gpt-4o-mini-audio-preview": {"input": 0.15, "cached_input": None, "output": 0.60},
    "o1": {"input": 15.00, "cached_input": 7.50, "output": 60.00},
    "o1-pro": {"input": 150.00, "cached_input": None, "output": 600.00},
    "o3-pro": {"input": 20.00, "cached_input": None, "output": 80.00},
    "o3": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "o3-deep-research": {"input": 10.00, "cached_input": 2.50, "output": 40.00},
    "o4-mini": {"input": 1.10, "cached_input": 0.275, "output": 4.40},
    "o4-mini-deep-research": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "o3-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40},
    "o1-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40},
    "gpt-5.1-codex-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "codex-mini-latest": {"input": 1.50, "cached_input": 0.375, "output": 6.00},
    "gpt-5-search-api": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-4o-mini-search-preview": {"input": 0.15, "cached_input": None, "output": 0.60},
    "gpt-4o-search-preview": {"input": 2.50, "cached_input": None, "output": 10.00},
    "computer-use-preview": {"input": 3.00, "cached_input": None, "output": 12.00},
    "gpt-image-1.5": {"input": 5.00, "cached_input": 1.25, "output": 10.00},
    "chatgpt-image-latest": {"input": 5.00, "cached_input": 1.25, "output": 10.00},
    "gpt-image-1": {"input": 5.00, "cached_input": 1.25, "output": None},
    "gpt-image-1-mini": {"input": 2.00, "cached_input": 0.20, "output": None},
}

# Image tokens
# Prices per 1M tokens
OPENAI_IMAGE_PRICES: Dict[str, Dict[str, Any]] = {
    "gpt-image-1.5": {"input": 8.00, "cached_input": 2.00, "output": 32.00},
    "chatgpt-image-latest": {"input": 8.00, "cached_input": 2.00, "output": 32.00},
    "gpt-image-1": {"input": 10.00, "cached_input": 2.50, "output": 40.00},
    "gpt-image-1-mini": {"input": 2.50, "cached_input": 0.25, "output": 8.00},
    "gpt-realtime": {"input": 5.00, "cached_input": 0.50, "output": None},
    "gpt-realtime-mini": {"input": 0.80, "cached_input": 0.08, "output": None},
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

# Video tokens
# Price per second
OPENAI_VIDEO_PRICES: Dict[str, Dict[str, Any]] = {
    "sora-2": {"Portrait: 720x1280 Landscape: 1280x720": 0.10},
    "sora-2-pro": {
        "Portrait: 720x1280 Landscape: 1280x720": 0.30,
        "Portrait: 1024x1792 Landscape: 1792x1024": 0.50,
    },
}

# Fine-tuning
# Prices per 1M tokens
OPENAI_FINE_TUNING_PRICES: Dict[str, Dict[str, Any]] = {
    "o4-mini-2025-04-16": {
        "training": 100.00,
        "input": 4.00,
        "cached_input": 1.00,
        "output": 16.00,
    },
    "o4-mini-2025-04-16-data-sharing": {
        "training": 100.00,
        "input": 2.00,
        "cached_input": 0.50,
        "output": 8.00,
    },
    "gpt-4.1-2025-04-14": {
        "training": 25.00,
        "input": 3.00,
        "cached_input": 0.75,
        "output": 12.00,
    },
    "gpt-4.1-mini-2025-04-14": {
        "training": 5.00,
        "input": 0.80,
        "cached_input": 0.20,
        "output": 3.20,
    },
    "gpt-4.1-nano-2025-04-14": {
        "training": 1.50,
        "input": 0.20,
        "cached_input": 0.05,
        "output": 0.80,
    },
    "gpt-4o-2024-08-06": {
        "training": 25.00,
        "input": 3.75,
        "cached_input": 1.875,
        "output": 15.00,
    },
    "gpt-4o-mini-2024-07-18": {
        "training": 3.00,
        "input": 0.30,
        "cached_input": 0.15,
        "output": 1.20,
    },
    "gpt-3.5-turbo": {
        "training": 8.00,
        "input": 3.00,
        "cached_input": None,
        "output": 6.00,
    },
    "davinci-002": {
        "training": 6.00,
        "input": 12.00,
        "cached_input": None,
        "output": 12.00,
    },
    "babbage-002": {
        "training": 0.40,
        "input": 1.60,
        "cached_input": None,
        "output": 1.60,
    },
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

# Image generation
# Prices per image
OPENAI_IMAGE_GENERATION_PRICES: Dict[str, Dict[str, Any]] = {
    "gpt-image-1.5": {
        "low": {"1024x1024": 0.009, "1024x1536": 0.013, "1536x1024": 0.013},
        "medium": {"1024x1024": 0.034, "1024x1536": 0.05, "1536x1024": 0.05},
        "high": {"1024x1024": 0.133, "1024x1536": 0.2, "1536x1024": 0.2},
    },
    "chatgpt-image-latest": {
        "low": {"1024x1024": 0.009, "1024x1536": 0.013, "1536x1024": 0.013},
        "medium": {"1024x1024": 0.034, "1024x1536": 0.05, "1536x1024": 0.05},
        "high": {"1024x1024": 0.133, "1024x1536": 0.2, "1536x1024": 0.2},
    },
    "gpt-image-1": {
        "low": {"1024x1024": 0.011, "1024x1536": 0.016, "1536x1024": 0.016},
        "medium": {"1024x1024": 0.042, "1024x1536": 0.063, "1536x1024": 0.063},
        "high": {"1024x1024": 0.167, "1024x1536": 0.25, "1536x1024": 0.25},
    },
    "gpt-image-1-mini": {
        "low": {"1024x1024": 0.005, "1024x1536": 0.006, "1536x1024": 0.006},
        "medium": {"1024x1024": 0.011, "1024x1536": 0.015, "1536x1024": 0.015},
        "high": {"1024x1024": 0.036, "1024x1536": 0.052, "1536x1024": 0.052},
    },
    "dalle-3": {
        "standard": {"1024x1024": 0.04, "1024x1792": 0.08, "1792x1024": 0.08},
        "hd": {"1024x1024": 0.08, "1024x1792": 0.12, "1792x1024": 0.12},
    },
    "dalle-2": {
        "standard": {"256x256": 0.016, "512x512": 0.018, "1024x1024": 0.02},
    },
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
OPENAI_LEGACY_PRICES: Dict[str, Dict[str, float]] = {
    "chatgpt-4o-latest": {"input": 5.00, "output": 15.00},
    "gpt-4-turbo-2024-04-09": {"input": 10.00, "output": 30.00},
    "gpt-4-0125-preview": {"input": 10.00, "output": 30.00},
    "gpt-4-1106-preview": {"input": 10.00, "output": 30.00},
    "gpt-4-1106-vision-preview": {"input": 10.00, "output": 30.00},
    "gpt-4-0613": {"input": 30.00, "output": 60.00},
    "gpt-4-0314": {"input": 30.00, "output": 60.00},
    "gpt-4-32k": {"input": 60.00, "output": 120.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-3.5-turbo-0125": {"input": 0.50, "output": 1.50},
    "gpt-3.5-turbo-1106": {"input": 1.00, "output": 2.00},
    "gpt-3.5-turbo-0613": {"input": 1.50, "output": 2.00},
    "gpt-3.5-0301": {"input": 1.50, "output": 2.00},
    "gpt-3.5-turbo-instruct": {"input": 1.50, "output": 2.00},
    "gpt-3.5-turbo-16k-0613": {"input": 3.00, "output": 4.00},
    "davinci-002": {"input": 2.00, "output": 2.00},
    "babbage-002": {"input": 0.40, "output": 0.40},
}
