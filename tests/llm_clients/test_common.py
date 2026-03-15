from pydantic import BaseModel
from kavalai.llm_clients.common import (
    create_model_call_stat,
    StreamContent,
)
from kavalai.agents.db import ModelCallStat


class SimpleModel(BaseModel):
    name: str


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


def test_stream_content_pydantic():
    sc = StreamContent(type="text", name="chunk", value="hello")
    assert sc.type == "text"
    assert sc.name == "chunk"
    assert sc.value == "hello"


from kavalai.llm_clients.common import fix_json


def test_fix_json_valid():
    data = '{"a": "hello"}'
    assert fix_json(data) == {"a": "hello"}


def test_fix_json_trailing_chars():
    data = '{"a": "hello"}...'
    # ensure_json should truncate the trailing '...'
    parsed = fix_json(data)
    assert parsed == {"a": "hello"}


def test_fix_json_incomplete():
    data = '{"a": "hello'
    # partial_json_parser's ensure_json should close it
    parsed = fix_json(data)
    assert parsed == {"a": "hello"}


def test_fix_json_leading_garbage():
    data = 'xx asdf{"a": "hello"}..'
    parsed = fix_json(data)
    assert parsed == {"a": "hello"}


def test_fix_json_complex_trailing():
    data = '{"a": "hello"} some garbage here'
    parsed = fix_json(data)
    assert parsed == {"a": "hello"}


def test_fix_json_multiple_trailing_braces():
    data = '{"a": "hello"}}'
    parsed = fix_json(data)
    assert parsed == {"a": "hello"}


def test_fix_json_trailing_comma():
    data = '{"a": "hello",}'
    parsed = fix_json(data)
    assert parsed == {"a": "hello"}


def test_fix_json_trailing_comma_partial():
    data = '{"a": "hello",'
    parsed = fix_json(data)
    assert parsed == {"a": "hello"}


def test_fix_json_broken_middle():
    data = '{"a": "hello", "b": }'
    parsed = fix_json(data)
    # It should at least return something that is a dict/list if possible
    assert isinstance(parsed, (dict, list))
