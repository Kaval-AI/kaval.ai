from kavalai.llm_clients.kwargs_mapper import LLMKWargsMapper


def test_openai_reasoning_and_stops_and_max_tokens_mapping():
    kwargs = {
        "reasoning": "high",
        "temperature": 0.2,
        "stop_sequences": ["END"],
        "max_tokens": 100,
    }
    mapped = LLMKWargsMapper.map("openai", "gpt-4o", kwargs)

    assert mapped.get("reasoning_effort") == "high"
    assert mapped.get("temperature") == 0.2
    assert mapped.get("stop") == ["END"]
    assert mapped.get("max_output_tokens") == 100
    assert "stop_sequences" not in mapped
    assert "max_tokens" not in mapped


def test_gemini_reasoning_level_and_stops_and_max_tokens_mapping():
    kwargs = {
        "reasoning": "medium",
        "temperature": 0.7,
        "stop": "#END",
        "max_tokens": 64,
        "presence_penalty": 0.1,  # should be stripped for gemini
    }
    mapped = LLMKWargsMapper.map("gemini", "gemini-2.0-flash", kwargs)

    assert mapped.get("thinking_level") == "medium"
    assert mapped.get("temperature") == 0.7
    assert mapped.get("stop_sequences") == ["#END"]
    assert mapped.get("max_output_tokens") == 64
    assert "presence_penalty" not in mapped


def test_gemini_reasoning_budget_mapping():
    kwargs = {"reasoning": 300}
    mapped = LLMKWargsMapper.map("gemini", "gemini-2.0-flash-thinking", kwargs)
    assert mapped.get("thinking_budget") == 300


def test_passthrough_existing_specific_keys():
    # If user already provided provider-specific key, do not override
    oa_kwargs = {"reasoning_effort": "low"}
    oa_mapped = LLMKWargsMapper.map("openai", "gpt-4o", oa_kwargs)
    assert oa_mapped.get("reasoning_effort") == "low"

    ge_kwargs = {"thinking_budget": 500}
    ge_mapped = LLMKWargsMapper.map("gemini", "gemini-2.0-flash-thinking", ge_kwargs)
    assert ge_mapped.get("thinking_budget") == 500


def test_priority_mapping():
    # OpenAI
    assert (
        LLMKWargsMapper.map("openai", "gpt-4o", {"priority": "high"}).get(
            "service_tier"
        )
        == "priority"
    )
    assert (
        LLMKWargsMapper.map("openai", "gpt-4o", {"priority": "low"}).get("service_tier")
        == "flex"
    )
    assert (
        LLMKWargsMapper.map("openai", "gpt-4o", {"priority": "normal"}).get(
            "service_tier"
        )
        == "default"
    )

    # Gemini
    assert (
        LLMKWargsMapper.map("gemini", "gemini-2.0-flash", {"priority": "high"}).get(
            "service_tier"
        )
        == "priority"
    )
    assert (
        LLMKWargsMapper.map("gemini", "gemini-2.0-flash", {"priority": "low"}).get(
            "service_tier"
        )
        == "flex"
    )
    assert (
        LLMKWargsMapper.map("gemini", "gemini-2.0-flash", {"priority": "normal"}).get(
            "service_tier"
        )
        is None
    )  # default for gemini is None
