from kavalai.prices.openai import (
    OPENAI_TEXT_PRICES,
    OPENAI_AUDIO_PRICES,
    OPENAI_FINE_TUNING_PRICES,
    OPENAI_TOOL_PRICES,
    OPENAI_AGENTKIT_PRICES,
    OPENAI_TRANSCRIPTION_SPEECH_TEXT_PRICES,
    OPENAI_TRANSCRIPTION_SPEECH_AUDIO_PRICES,
    OPENAI_TRANSCRIPTION_SPEECH_OTHER_PRICES,
    OPENAI_EMBEDDING_PRICES,
    OPENAI_LEGACY_PRICES,
)


def test_openai_text_prices():
    assert OPENAI_TEXT_PRICES["gpt-5.2"].input.price_per_1m == 1.75
    assert OPENAI_TEXT_PRICES["gpt-4o"].input.price_per_1m == 2.50
    assert OPENAI_TEXT_PRICES["o1"].output.price_per_1m == 60.00
    assert OPENAI_TEXT_PRICES["gpt-image-1"].output.price_per_1m == 0.0


def test_openai_audio_prices():
    assert OPENAI_AUDIO_PRICES["gpt-realtime"]["input"] == 32.00
    assert OPENAI_AUDIO_PRICES["gpt-4o-audio-preview"]["output"] == 80.00


def test_openai_fine_tuning_prices():
    assert OPENAI_FINE_TUNING_PRICES["o4-mini-2025-04-16"].training == 100.00
    assert OPENAI_FINE_TUNING_PRICES["gpt-3.5-turbo"].output.price_per_1m == 6.00


def test_openai_tool_prices():
    assert OPENAI_TOOL_PRICES["code_interpreter"]["1 GB"] == 0.03
    assert OPENAI_TOOL_PRICES["file_search_storage"] == 0.10
    assert OPENAI_TOOL_PRICES["web_search"] == 10.00


def test_openai_agentkit_prices():
    assert OPENAI_AGENTKIT_PRICES["chatkit_storage"] == 0.10


def test_openai_transcription_speech_prices():
    assert OPENAI_TRANSCRIPTION_SPEECH_TEXT_PRICES["gpt-4o-mini-tts"]["input"] == 0.60
    assert (
        OPENAI_TRANSCRIPTION_SPEECH_AUDIO_PRICES["gpt-4o-mini-tts"]["output"] == 12.00
    )
    assert OPENAI_TRANSCRIPTION_SPEECH_OTHER_PRICES["whisper"] == 0.006


def test_openai_embedding_prices():
    assert OPENAI_EMBEDDING_PRICES["text-embedding-3-small"]["standard"] == 0.02
    assert OPENAI_EMBEDDING_PRICES["text-embedding-3-small"]["batch"] == 0.01


def test_openai_legacy_prices():
    assert OPENAI_LEGACY_PRICES["gpt-4-turbo-2024-04-09"].input.price_per_1m == 10.00
    assert OPENAI_LEGACY_PRICES["gpt-3.5-turbo"].output.price_per_1m == 1.50
