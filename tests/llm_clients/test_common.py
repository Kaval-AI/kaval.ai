import pytest
from pydantic import BaseModel
from kavalai.llm_clients.common import (
    create_model_call_stat,
    normalize_embeddings,
    StreamContent,
)
from kavalai.agents.db import ModelCallStat


class SimpleModel(BaseModel):
    name: str


def test_normalize_embeddings():
    embeddings = [[1.0, 1.0], [0.0, 0.0], [3.0, 4.0]]
    normalized = normalize_embeddings(embeddings)

    # [1.0, 1.0] -> norm is sqrt(2), so [1/sqrt(2), 1/sqrt(2)]
    assert pytest.approx(sum(x * x for x in normalized[0])) == 1.0
    # [0.0, 0.0] -> norm is 0, should remain [0.0, 0.0]
    assert normalized[1] == [0.0, 0.0]
    # [3.0, 4.0] -> norm is 5, so [0.6, 0.8]
    assert normalized[2] == [0.6, 0.8]
    assert pytest.approx(sum(x * x for x in normalized[2])) == 1.0


def test_create_model_call_stat_basic():
    stat = create_model_call_stat(
        call_type="llm",
        model="test-model",
        duration=1.5,
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
        call_type="llm", model="test-model", duration=1.0, response_data="just a string"
    )
    assert stat.response_data == "just a string"


def test_create_model_call_stat_none_response():
    stat = create_model_call_stat(
        call_type="llm", model="test-model", duration=1.0, response_data=None
    )
    assert stat.response_data is None


def test_stream_content_pydantic():
    sc = StreamContent(type="text", name="chunk", value="hello")
    assert sc.type == "text"
    assert sc.name == "chunk"
    assert sc.value == "hello"
