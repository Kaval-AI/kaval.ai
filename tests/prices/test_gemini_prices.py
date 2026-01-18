from kavalai.prices.gemini import (
    GEMINI_PRICES,
    GEMINI_IMAGE_GENERATION_PRICES,
    GEMINI_VIDEO_GENERATION_PRICES,
)


def test_gemini_prices():
    # Test gemini-3-pro-preview
    assert GEMINI_PRICES["gemini-3-pro-preview"]["text"]["input"]["<=200k"] == 1.00
    assert GEMINI_PRICES["gemini-3-pro-preview"]["text"]["output"][">200k"] == 9.00
    assert GEMINI_PRICES["gemini-3-pro-preview"]["grounding"]["google_search"] == 14.00

    # Test gemini-3-flash-preview
    assert GEMINI_PRICES["gemini-3-flash-preview"]["text_image_video"]["input"] == 0.25
    assert GEMINI_PRICES["gemini-3-flash-preview"]["audio"]["input"] == 0.50
    assert (
        GEMINI_PRICES["gemini-3-flash-preview"][
            "context_caching_storage_per_1M_per_hour"
        ]
        == 1.00
    )

    # Test gemini-2.5-pro
    assert GEMINI_PRICES["gemini-2.5-pro"]["text"]["input"]["<=200k"] == 0.625
    assert GEMINI_PRICES["gemini-2.5-pro"]["text"]["output"][">200k"] == 7.50
    assert GEMINI_PRICES["gemini-2.5-pro"]["grounding"]["google_search"] == 35.00

    # Test gemini-2.0-flash
    assert GEMINI_PRICES["gemini-2.0-flash"]["text_image_video"]["input"] == 0.05
    assert GEMINI_PRICES["gemini-2.0-flash"]["audio"]["context_caching"] == 0.175
    assert GEMINI_PRICES["gemini-2.0-flash"]["image_generation_per_image"] == 0.0195

    # Test computer use
    assert (
        GEMINI_PRICES["gemini-2.5-computer-use-preview-10-2025"]["input"][">200k"]
        == 2.50
    )


def test_gemini_image_generation_prices():
    assert GEMINI_IMAGE_GENERATION_PRICES["imagen-4.0-fast-generate-001"] == 0.02
    assert GEMINI_IMAGE_GENERATION_PRICES["imagen-3.0-generate-002"] == 0.03


def test_gemini_video_generation_prices():
    assert GEMINI_VIDEO_GENERATION_PRICES["veo-3.1-generate-preview"]["4k"] == 0.60
    assert GEMINI_VIDEO_GENERATION_PRICES["veo-3.0-fast-generate-001"] == 0.15
    assert GEMINI_VIDEO_GENERATION_PRICES["veo-2.0-generate-001"] == 0.35
