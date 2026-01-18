from typing import Dict, Any

# Prices per 1M tokens in USD unless otherwise specified
GEMINI_PRICES: Dict[str, Dict[str, Any]] = {
    "gemini-3-pro-preview": {
        "text": {
            "input": {"<=200k": 1.00, ">200k": 2.00},
            "output": {"<=200k": 6.00, ">200k": 9.00},
            "context_caching": {
                "<=200k": 0.20,
                ">200k": 0.40,
                "storage_per_1M_per_hour": 4.50,
            },
        },
        "grounding": {"google_search": 14.00},  # per 1k queries
    },
    "gemini-3-flash-preview": {
        "text_image_video": {
            "input": 0.25,
            "output": 1.50,
            "context_caching": 0.05,
        },
        "audio": {
            "input": 0.50,
            "output": 1.50,  # Matches text output price
            "context_caching": 0.10,
        },
        "context_caching_storage_per_1M_per_hour": 1.00,
        "grounding": {"google_search": 14.00},  # per 1k queries
    },
    "gemini-3-pro-image-preview": {
        "text": {
            "input": 1.00,
            "output": 6.00,
        },
        "image": {
            "input_per_image": 0.0011,  # 560 tokens
            "output_1k_2k": 0.067,
            "output_4k": 0.12,
        },
    },
    "gemini-2.5-pro": {
        "text": {
            "input": {"<=200k": 0.625, ">200k": 1.25},
            "output": {"<=200k": 5.00, ">200k": 7.50},
            "context_caching": {
                "<=200k": 0.125,
                ">200k": 0.25,
                "storage_per_1M_per_hour": 4.50,
            },
        },
        "grounding": {"google_search": 35.00},  # per 1k grounded prompts
    },
    "gemini-2.5-flash": {
        "text_image_video": {
            "input": 0.15,
            "output": 1.25,
            "context_caching": 0.03,
        },
        "audio": {
            "input": 0.50,
            "output": 1.25,
            "context_caching": 0.10,
        },
        "context_caching_storage_per_1M_per_hour": 1.00,
        "grounding": {"google_search": 35.00},  # per 1k grounded prompts
    },
    "gemini-2.5-flash-preview-09-2025": {
        "text_image_video": {
            "input": 0.15,
            "output": 1.25,
            "context_caching": 0.03,
        },
        "audio": {
            "input": 0.50,
            "output": 1.25,
            "context_caching": 0.10,
        },
        "context_caching_storage_per_1M_per_hour": 1.00,
        "grounding": {"google_search": 35.00},  # per 1k grounded prompts
    },
    "gemini-2.5-flash-lite": {
        "text_image_video": {
            "input": 0.05,
            "output": 0.20,
            "context_caching": 0.01,
        },
        "audio": {
            "input": 0.15,
            "output": 0.20,
            "context_caching": 0.03,
        },
        "context_caching_storage_per_1M_per_hour": 1.00,
        "grounding": {"google_search": 35.00},  # per 1k grounded prompts
    },
    "gemini-2.5-flash-lite-preview-09-2025": {
        "text_image_video": {
            "input": 0.05,
            "output": 0.20,
            "context_caching": 0.01,
        },
        "audio": {
            "input": 0.15,
            "output": 0.20,
            "context_caching": 0.03,
        },
        "context_caching_storage_per_1M_per_hour": 1.00,
        "grounding": {"google_search": 35.00},  # per 1k grounded prompts
    },
    "gemini-2.5-flash-native-audio-preview-12-2025": {
        "text": {
            "input": 0.50,
            "output": 2.00,
        },
        "audio_video": {
            "input": 3.00,
            "output_audio": 12.00,
        },
    },
    "gemini-2.5-flash-image": {
        "text_image": {
            "input": 0.15,
            "output_per_image": 0.0195,
        }
    },
    "gemini-2.5-flash-preview-tts": {
        "text_input": 0.25,
        "audio_output": 5.00,
    },
    "gemini-2.5-pro-preview-tts": {
        "text_input": 0.50,
        "audio_output": 10.00,
    },
    "gemini-2.0-flash": {
        "text_image_video": {
            "input": 0.05,
            "output": 0.20,
            "context_caching": 0.025,
        },
        "audio": {
            "input": 0.35,
            "output": 0.20,
            "context_caching": 0.175,
        },
        "context_caching_storage_per_1M_per_hour": 1.00,
        "image_generation_per_image": 0.0195,
        "grounding": {"google_search": 35.00},  # per 1k grounded prompts
    },
    "gemini-2.0-flash-lite": {
        "input": 0.0375,
        "output": 0.15,
    },
    "gemini-embedding-001": {
        "input": 0.075,
    },
    "gemini-2.5-computer-use-preview-10-2025": {
        "input": {"<=200k": 1.25, ">200k": 2.50},
        "output": {"<=200k": 10.00, ">200k": 15.00},
    },
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
