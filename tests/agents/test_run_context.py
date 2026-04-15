import pytest
from uuid import uuid4
from pydantic import BaseModel
from unittest.mock import AsyncMock

from kavalai.agents.run_context import RunContext
from kavalai.agents.workflow_model import PythonTask, ArgumentInfo


class MockModel(BaseModel):
    field: str


@pytest.mark.asyncio
async def test_resolve_context_value():
    rc = RunContext(data={"user": {"name": "Alice"}, "items": [1, 2, 3]})

    assert rc.resolve_context_value("user.name") == "Alice"
    assert rc.resolve_context_value("items.0") == 1
    assert rc.resolve_context_value("nonexistent") is None


@pytest.mark.asyncio
async def test_resolve_input_info_literal():
    rc = RunContext()
    info = ArgumentInfo(type="literal", value="hello")
    assert await rc.resolve_input_info(info) == "hello"


@pytest.mark.asyncio
async def test_resolve_input_info_context():
    rc = RunContext(data={"input": {"text": "world"}})

    # Using value as path
    info = ArgumentInfo(type="context", value="input.text")
    assert await rc.resolve_input_info(info) == "world"

    # Using name as path
    info = ArgumentInfo(type="context", name="input.text")
    assert await rc.resolve_input_info(info) == "world"

    # Path is None
    info = ArgumentInfo(type="context")
    assert await rc.resolve_input_info(info) is None


@pytest.mark.asyncio
async def test_resolve_input_info_history():
    session_id = uuid4()
    mock_service = AsyncMock()
    mock_service.get_history_value.return_value = "history_val"

    rc = RunContext(session_id=session_id, agent_service=mock_service)

    # Using value
    info = ArgumentInfo(type="history", value="some_key")
    assert await rc.resolve_input_info(info) == "history_val"
    mock_service.get_history_value.assert_called_with(session_id, "some_key")

    # Using name
    info = ArgumentInfo(type="history", name="other_key")
    assert await rc.resolve_input_info(info) == "history_val"
    mock_service.get_history_value.assert_called_with(session_id, "other_key")


@pytest.mark.asyncio
async def test_resolve_input_info_history_missing_service(caplog):
    rc = RunContext(session_id=uuid4())
    info = ArgumentInfo(type="history", value="key")
    assert await rc.resolve_input_info(info) is None
    assert (
        "Cannot load from history for key: agent_service or session_id not set"
        in caplog.text
    )


@pytest.mark.asyncio
async def test_prepare_tool_inputs():
    rc = RunContext(data={"val": "from_context"})
    task = PythonTask(
        name="test_task",
        python_tool="test_tool",
        inputs={
            "lit": ArgumentInfo(type="literal", value="fixed"),
            "ctx": ArgumentInfo(type="context", value="val"),
            "implicit": ArgumentInfo(type="context"),  # should use name "implicit"
        },
        output="output",
    )
    # Patch implicit context to match data
    rc.data["implicit"] = "implicit_val"

    inputs = await rc.prepare_tool_inputs(task)
    assert inputs == {"lit": "fixed", "ctx": "from_context", "implicit": "implicit_val"}


@pytest.mark.asyncio
async def test_prepare_tool_inputs_with_pydantic_model():
    rc = RunContext()
    model_instance = MockModel(field="test")
    task = PythonTask(
        name="test_task",
        python_tool="test_tool",
        inputs={"mod": ArgumentInfo(type="literal", value=model_instance)},
        output="output",
    )

    inputs = await rc.prepare_tool_inputs(task)
    assert inputs["mod"] == {"field": "test"}


@pytest.mark.asyncio
async def test_evaluate_condition_basic_operators():
    rc = RunContext(data={"a": 10, "b": 20, "s": "hello world"})

    # eq
    assert await rc.evaluate_condition({"eq": [10, 10]}) is True
    assert (
        await rc.evaluate_condition({"eq": [{"type": "context", "value": "a"}, 10]})
        is True
    )
    assert await rc.evaluate_condition({"eq": [10, 20]}) is False

    # not_eq
    assert await rc.evaluate_condition({"not_eq": [10, 20]}) is True

    # gt, gte, lt, lte
    assert await rc.evaluate_condition({"gt": [20, 10]}) is True
    assert await rc.evaluate_condition({"gte": [20, 20]}) is True
    assert await rc.evaluate_condition({"lt": [10, 20]}) is True
    assert await rc.evaluate_condition({"lte": [10, 10]}) is True

    # contains
    assert await rc.evaluate_condition({"contains": ["hello world", "hello"]}) is True
    assert (
        await rc.evaluate_condition(
            {"contains": [{"type": "context", "value": "s"}, "world"]}
        )
        is True
    )
    assert await rc.evaluate_condition({"contains": [None, "foo"]}) is False

    # is_null
    assert (
        await rc.evaluate_condition({"is_null": {"type": "context", "value": "a"}})
        is False
    )
    rc.data["n"] = None
    assert (
        await rc.evaluate_condition({"is_null": {"type": "context", "value": "n"}})
        is True
    )
    assert await rc.evaluate_condition({"is_null": None}) is True
    assert await rc.evaluate_condition({"is_null": 10}) is False

    # is_not_null
    assert (
        await rc.evaluate_condition({"is_not_null": {"type": "context", "value": "a"}})
        is True
    )
    assert (
        await rc.evaluate_condition({"is_not_null": {"type": "context", "value": "n"}})
        is False
    )
    assert await rc.evaluate_condition({"is_not_null": None}) is False
    assert await rc.evaluate_condition({"is_not_null": 10}) is True

    # is_true
    assert await rc.evaluate_condition({"is_true": True}) is True
    assert await rc.evaluate_condition({"is_true": False}) is False
    assert await rc.evaluate_condition({"is_true": 1}) is True
    assert await rc.evaluate_condition({"is_true": 0}) is False
    assert (
        await rc.evaluate_condition({"is_true": {"type": "context", "value": "a"}})
        is True
    )
    rc.data["f"] = False
    assert (
        await rc.evaluate_condition({"is_true": {"type": "context", "value": "f"}})
        is False
    )

    # len
    assert await rc.evaluate_condition({"len": [[1, 2, 3], 3]}) is True
    assert await rc.evaluate_condition({"len": [[1, 2, 3], 2]}) is False
    assert await rc.evaluate_condition({"len": ["hello", 5]}) is True
    rc.data["l"] = [1, 2]
    assert (
        await rc.evaluate_condition({"len": [{"type": "context", "value": "l"}, 2]})
        is True
    )
    assert await rc.evaluate_condition({"len": [None, 0]}) is False

    # Combination: size (using .length in path) greater than length of an array
    rc.data["arr"] = [1, 2, 3, 4, 5]
    rc.data["limit"] = 3
    assert (
        await rc.evaluate_condition(
            {"gt": [{"type": "context", "value": "arr.length"}, 3]}
        )
        is True
    )
    assert (
        await rc.evaluate_condition(
            {
                "gt": [
                    {"type": "context", "value": "arr.length"},
                    {"type": "context", "value": "limit"},
                ]
            }
        )
        is True
    )

    # Combination: size less than length
    assert (
        await rc.evaluate_condition(
            {"lt": [{"type": "context", "value": "arr.length"}, 10]}
        )
        is True
    )
    assert (
        await rc.evaluate_condition(
            {"lt": [{"type": "context", "value": "arr.length"}, 3]}
        )
        is False
    )

    # Complex combination with all/any
    assert (
        await rc.evaluate_condition(
            {
                "all": [
                    {"gt": [{"type": "context", "value": "arr.length"}, 2]},
                    {"contains": [{"type": "context", "value": "s"}, "hello"]},
                ]
            }
        )
        is True
    )

    # Nested path with length
    rc.data["user"] = {"items": ["a", "b"]}
    assert (
        await rc.evaluate_condition(
            {"eq": [{"type": "context", "value": "user.items.length"}, 2]}
        )
        is True
    )
    assert (
        await rc.evaluate_condition(
            {"gte": [{"type": "context", "value": "user.items.length"}, 2]}
        )
        is True
    )

    # Complex nested logical: NOT (ANY (ALL(...), ALL(...)))
    # (a > 5 AND s contains "foo") OR (b == 20 AND arr.length < 3)
    # We want NOT of that.
    rc.data["a"] = 10
    rc.data["s"] = "hello world"  # does NOT contain "foo"
    rc.data["b"] = 10  # is NOT 20
    rc.data["arr"] = [1, 2, 3, 4, 5]  # length 5 is NOT < 3
    # Inner ALL 1: False
    # Inner ALL 2: False
    # ANY: False
    # NOT ANY: True
    cond = {
        "not": {
            "any": [
                {
                    "all": [
                        {"gt": [{"type": "context", "value": "a"}, 5]},
                        {"contains": [{"type": "context", "value": "s"}, "foo"]},
                    ]
                },
                {
                    "all": [
                        {"eq": [{"type": "context", "value": "b"}, 20]},
                        {"lt": [{"type": "context", "value": "arr.length"}, 3]},
                    ]
                },
            ]
        }
    }
    assert await rc.evaluate_condition(cond) is True

    # Now make it False by making one of the inner ALLs True
    rc.data["s"] = "foo bar"
    # Inner ALL 1: (10 > 5 AND "foo bar" contains "foo") -> True
    # ANY: True
    # NOT ANY: False
    assert await rc.evaluate_condition(cond) is False

    # Check contains with list context
    rc.data["allowed_users"] = ["alice", "bob"]
    assert (
        await rc.evaluate_condition(
            {"contains": [{"type": "context", "value": "allowed_users"}, "alice"]}
        )
        is True
    )
    assert (
        await rc.evaluate_condition(
            {"contains": [{"type": "context", "value": "allowed_users"}, "charlie"]}
        )
        is False
    )


@pytest.mark.asyncio
async def test_evaluate_condition_logical():
    rc = RunContext()

    # all
    assert (
        await rc.evaluate_condition({"all": [{"eq": [1, 1]}, {"eq": [2, 2]}]}) is True
    )
    assert (
        await rc.evaluate_condition({"all": [{"eq": [1, 1]}, {"eq": [2, 3]}]}) is False
    )

    # any
    assert (
        await rc.evaluate_condition({"any": [{"eq": [1, 2]}, {"eq": [2, 2]}]}) is True
    )
    assert (
        await rc.evaluate_condition({"any": [{"eq": [1, 2]}, {"eq": [2, 3]}]}) is False
    )

    # not
    assert await rc.evaluate_condition({"not": {"eq": [1, 2]}}) is True
    assert await rc.evaluate_condition({"not": {"eq": [1, 1]}}) is False


@pytest.mark.asyncio
async def test_evaluate_condition_errors():
    rc = RunContext()

    with pytest.raises(ValueError, match="requires a list of 2 operands"):
        await rc.evaluate_condition({"eq": [1]})

    with pytest.raises(ValueError, match="'all' requires a list of conditions"):
        await rc.evaluate_condition({"all": {"eq": [1, 1]}})

    with pytest.raises(ValueError, match="'any' requires a list of conditions"):
        await rc.evaluate_condition({"any": {"eq": [1, 1]}})

    with pytest.raises(
        ValueError, match="'not' requires a single condition dictionary"
    ):
        await rc.evaluate_condition({"not": []})


@pytest.mark.asyncio
async def test_evaluate_condition_complex_none():
    rc = RunContext(data={"a": None, "b": 10})

    # all with None
    assert (
        await rc.evaluate_condition(
            {"all": [{"is_null": {"type": "context", "value": "a"}}, {"eq": [10, 10]}]}
        )
        is True
    )
    assert (
        await rc.evaluate_condition(
            {
                "all": [
                    {"is_not_null": {"type": "context", "value": "a"}},
                    {"eq": [10, 10]},
                ]
            }
        )
        is False
    )

    # any with None
    assert (
        await rc.evaluate_condition(
            {
                "any": [
                    {"is_not_null": {"type": "context", "value": "a"}},
                    {"eq": [10, 10]},
                ]
            }
        )
        is True
    )
    assert (
        await rc.evaluate_condition(
            {
                "any": [
                    {"is_not_null": {"type": "context", "value": "a"}},
                    {"eq": [10, 20]},
                ]
            }
        )
        is False
    )

    # complex expression
    # (a is null AND b == 10) OR (b == 20)
    cond = {
        "any": [
            {
                "all": [
                    {"is_null": {"type": "context", "value": "a"}},
                    {"eq": [{"type": "context", "value": "b"}, 10]},
                ]
            },
            {"eq": [{"type": "context", "value": "b"}, 20]},
        ]
    }
    assert await rc.evaluate_condition(cond) is True

    rc.data["b"] = 20
    assert await rc.evaluate_condition(cond) is True

    rc.data["a"] = "not null"
    rc.data["b"] = 10
    assert await rc.evaluate_condition(cond) is False


@pytest.mark.asyncio
async def test_evaluate_condition_short_circuit():
    rc = RunContext()

    # 'all' short-circuit: second condition is invalid (not a list of 2),
    # but it shouldn't be reached if the first one is False.
    assert await rc.evaluate_condition({"all": [{"eq": [1, 2]}, {"eq": [1]}]}) is False

    # If first is True, second IS reached and raises ValueError
    with pytest.raises(ValueError, match="requires a list of 2 operands"):
        await rc.evaluate_condition({"all": [{"eq": [1, 1]}, {"eq": [1]}]})

    # 'any' short-circuit: second condition is invalid,
    # but it shouldn't be reached if the first one is True.
    assert await rc.evaluate_condition({"any": [{"eq": [1, 1]}, {"eq": [1]}]}) is True

    # If first is False, second IS reached and raises ValueError
    with pytest.raises(ValueError, match="requires a list of 2 operands"):
        await rc.evaluate_condition({"any": [{"eq": [1, 2]}, {"eq": [1]}]})


@pytest.mark.asyncio
async def test_evaluate_condition_unknown_key():
    rc = RunContext()
    # If the key is not an operator, 'all', 'any', or 'not', it should fall through to return True (line 135)
    assert await rc.evaluate_condition({"unknown": "value"}) is True


@pytest.mark.asyncio
async def test_evaluate_condition_empty():
    rc = RunContext()
    assert await rc.evaluate_condition({}) is True
    assert await rc.evaluate_condition(None) is True
