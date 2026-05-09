import asyncio
import json
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from kavalai.agents.db import ModelCallStat
from kavalai.llm_clients.common import (
    create_model_call_stat,
    StreamContent,
    Streamer,
    get_model_name,
    safe_parse_json,
)


class SimpleModel(BaseModel):
    name: str


# --- safe_parse_json tests ---

def test_safe_parse_json_valid():
    data = '{"a": "hello"}'
    assert safe_parse_json(data) == {"a": "hello"}
    assert safe_parse_json("[1, 2, 3]") == [1, 2, 3]


def test_safe_parse_json_trailing_chars():
    data = '{"a": "hello"}...'
    # ensure_json should truncate the trailing '...'
    parsed = safe_parse_json(data)
    assert parsed == {"a": "hello"}


def test_safe_parse_json_incomplete():
    data = '{"a": "hello'
    # partial_json_parser's ensure_json should close it
    parsed = safe_parse_json(data)
    assert parsed == {"a": "hello"}


def test_safe_parse_json_leading_garbage():
    data = 'xx asdf{"a": "hello"}..'
    parsed = safe_parse_json(data)
    assert parsed == {"a": "hello"}


def test_safe_parse_json_complex_trailing():
    data = '{"a": "hello"} some garbage here'
    parsed = safe_parse_json(data)
    assert parsed == {"a": "hello"}


def test_safe_parse_json_multiple_trailing_braces():
    data = '{"a": "hello"}}'
    parsed = safe_parse_json(data)
    assert parsed == {"a": "hello"}


def test_safe_parse_json_trailing_comma():
    data = '{"a": "hello",}'
    parsed = safe_parse_json(data)
    assert parsed == {"a": "hello"}
    assert safe_parse_json("[1, 2,]") == [1, 2]


def test_safe_parse_json_trailing_comma_partial():
    data = '{"a": "hello",'
    parsed = safe_parse_json(data)
    assert parsed == {"a": "hello"}


def test_safe_parse_json_no_structure():
    assert safe_parse_json("123") == 123
    assert safe_parse_json("true") is True
    assert safe_parse_json("null") is None
    assert safe_parse_json("not json at all") == {}


def test_safe_parse_json_extra_data_msg():
    # specifically trigger the "Extra data" branch
    data = '{"a": 1} {"b": 2}'
    assert safe_parse_json(data) == {"a": 1}
    with patch("json.loads") as mock_loads:
        mock_loads.side_effect = [
            json.JSONDecodeError("Extra data", '{"a": 1} {', 8),
            Exception("inner failure"),
            {"a": 1} # ensure_json result (mocked via json.loads)
        ]
        assert safe_parse_json('{"a": 1} {') == {"a": 1}


def test_safe_parse_json_ensure_json_fails():
    with patch("kavalai.llm_clients.common.ensure_json") as mock_ensure:
        mock_ensure.side_effect = Exception("error")
        assert safe_parse_json('{"a": 1} broken') == {"a": 1}


def test_safe_parse_json_ensure_json_twice_fails():
    with patch("kavalai.llm_clients.common.ensure_json") as mock_ensure:
        mock_ensure.side_effect = [Exception("error1"), Exception("error2")]
        assert safe_parse_json('{"a": 1} broken') == {"a": 1}


def test_safe_parse_json_fallback_slicing():
    assert safe_parse_json('{"a": 1} ]') == {"a": 1}
    assert safe_parse_json('[1, 2] }') == [1, 2]


def test_safe_parse_json_fallback_fails_json_loads():
    with patch("kavalai.llm_clients.common.ensure_json") as mock_ensure:
        mock_ensure.side_effect = Exception("error")
        try:
            json.loads('{"a": 1} broken')
        except json.JSONDecodeError as e:
            print(f"DEBUG: {e.msg}")

        assert safe_parse_json('{"a": 1} broken') == {"a": 1}


def test_safe_parse_json_fallback_rfind_exception():
    # Trigger Exception in the rfind loop
    with patch("kavalai.llm_clients.common.ensure_json") as mock_ensure:
        mock_ensure.side_effect = Exception("ensure_json failed")
        
        # We need something where rfind finds a char, but json.loads(subset) fails.
        # And ensure_json also fails.
        
        # Let's mock json.loads to fail inside the fallback loop.
        with patch("json.loads") as mock_loads:
            mock_loads.side_effect = [
                json.JSONDecodeError("fail", "", 0), # initial load
                Exception("ensure failed"),          # ensure_json -> json.loads
                Exception("fallback failed")         # fallback loop -> json.loads
            ]
            assert safe_parse_json('{"a": 1}') == {}


def test_safe_parse_json_really_all_fails():
    with patch("kavalai.llm_clients.common.ensure_json") as mock_ensure:
        mock_ensure.side_effect = Exception("error")
        assert safe_parse_json('{ no braces here') == {}


# --- StreamContent tests ---

def test_stream_content_pydantic():
    sc = StreamContent(type="text", name="chunk", value="hello")
    assert sc.type == "text"
    assert sc.name == "chunk"
    assert sc.value == "hello"


# --- Streamer tests ---

@pytest.mark.asyncio
async def test_streamer_partial():
    queue = asyncio.Queue()
    streamer = Streamer(name="test", queue=queue)
    await streamer.stream_partial(value="chunk1")
    res = await queue.get()
    parsed = json.loads(res)
    assert parsed["type"] == "partial"
    assert parsed["name"] == "test"
    assert parsed["value"] == "chunk1"


@pytest.mark.asyncio
async def test_streamer_complete():
    queue = asyncio.Queue()
    streamer = Streamer(name="test", queue=queue)
    await streamer.stream_complete(value="final", name="override")
    res = await queue.get()
    parsed = json.loads(res)
    assert parsed["type"] == "complete"
    assert parsed["name"] == "override"
    assert parsed["value"] == "final"


# --- get_model_name tests ---

def test_get_model_name():
    assert get_model_name("openai/gpt-4") == "gpt-4"
    assert get_model_name("gpt-3.5-turbo") == "gpt-3.5-turbo"


# --- create_model_call_stat tests ---

def test_create_model_call_stat_basic():
    stat = create_model_call_stat(
        call_type="llm",
        model="test-model",
        duration_sections=1.5,
        prompt_tokens=10,
        completion_tokens=20,
        response_data={"foo": "bar"},
    )
    assert isinstance(stat, ModelCallStat)
    assert stat.call_type == "llm"
    assert stat.model == "test-model"
    assert float(stat.duration_seconds) == 1.5
    assert stat.prompt_tokens == 10
    assert stat.completion_tokens == 20
    assert stat.total_tokens == 30
    assert stat.response_data == {"foo": "bar"}
    assert stat.response_code == 200


def test_create_model_call_stat_string_response():
    stat = create_model_call_stat(
        call_type="llm",
        model="test-model",
        duration_sections=1.0,
        response_data="just a string",
    )
    assert stat.response_data == "just a string"


def test_create_model_call_stat_none_response():
    stat = create_model_call_stat(
        call_type="llm", model="test-model", duration_sections=1.0, response_data=None
    )
    assert stat.response_data is None


def test_create_model_call_stat_tokens():
    stat = create_model_call_stat(
        call_type="llm",
        model="test",
        duration_sections=1.0,
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=None,
    )
    assert stat.total_tokens == 30
