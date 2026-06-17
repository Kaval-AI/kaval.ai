from pydantic import BaseModel
from kavalai.agents.utils import to_plain


class SimpleModel(BaseModel):
    name: str
    value: int


class NestedModel(BaseModel):
    simple: SimpleModel
    tags: list[str]


def test_to_plain_primitives():
    assert to_plain(1) == 1
    assert to_plain("test") == "test"
    assert to_plain(True) is True
    assert to_plain(None) is None


def test_to_plain_dict():
    data = {"a": 1, "b": {"c": 2}}
    assert to_plain(data) == data


def test_to_plain_list():
    data = [1, [2, 3], {"a": 4}]
    assert to_plain(data) == data


def test_to_plain_tuple():
    data = (1, 2)
    assert to_plain(data) == [1, 2]


def test_to_plain_base_model():
    model = SimpleModel(name="test", value=123)
    assert to_plain(model) == {"name": "test", "value": 123}


def test_to_plain_nested_structures():
    model = NestedModel(simple=SimpleModel(name="inner", value=1), tags=["a", "b"])
    data = {"model": model, "list": [model, {"direct": 1}], "plain": "value"}

    expected = {
        "model": {"simple": {"name": "inner", "value": 1}, "tags": ["a", "b"]},
        "list": [
            {"simple": {"name": "inner", "value": 1}, "tags": ["a", "b"]},
            {"direct": 1},
        ],
        "plain": "value",
    }

    assert to_plain(data) == expected
