from kavalai.prices.gemini import (
    GEMINI_PRICES,
    GEMINI_IMAGE_GENERATION_PRICES,
    GEMINI_VIDEO_GENERATION_PRICES,
)


def test_gemini_prices():
    # Test gemini-3-pro-preview
    assert GEMINI_PRICES["gemini-3-pro-preview"].input.tiered["<=200k"] == 1.00
    assert GEMINI_PRICES["gemini-3-pro-preview"].output.tiered[">200k"] == 9.00
    assert GEMINI_PRICES["gemini-3-pro-preview"].cached_input.tiered["<=200k"] == 0.20

    # Test gemini-3-flash-preview
    assert GEMINI_PRICES["gemini-3-flash-preview"].input.price_per_1m == 0.25
    assert GEMINI_PRICES["gemini-3-flash-preview"].output.price_per_1m == 1.50

    # Test gemini-2.5-pro
    assert GEMINI_PRICES["gemini-2.5-pro"].input.tiered["<=200k"] == 0.625
    assert GEMINI_PRICES["gemini-2.5-pro"].output.tiered[">200k"] == 7.50

    # Test gemini-2.0-flash
    assert GEMINI_PRICES["gemini-2.0-flash"].input.price_per_1m == 0.05
    assert GEMINI_PRICES["gemini-2.0-flash"].output.price_per_1m == 0.20

    # Test computer use
    assert (
        GEMINI_PRICES["gemini-2.5-computer-use-preview-10-2025"].input.tiered[">200k"]
        == 2.50
    )


def test_gemini_image_generation_prices():
    assert GEMINI_IMAGE_GENERATION_PRICES["imagen-4.0-fast-generate-001"] == 0.02
    assert GEMINI_IMAGE_GENERATION_PRICES["imagen-3.0-generate-002"] == 0.03


def test_gemini_video_generation_prices():
    assert GEMINI_VIDEO_GENERATION_PRICES["veo-3.1-generate-preview"]["4k"] == 0.60
    assert GEMINI_VIDEO_GENERATION_PRICES["veo-3.0-fast-generate-001"] == 0.15
    assert GEMINI_VIDEO_GENERATION_PRICES["veo-2.0-generate-001"] == 0.35
