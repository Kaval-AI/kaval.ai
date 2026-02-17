import pytest
from uuid import uuid4
from pydantic import BaseModel
from unittest.mock import AsyncMock

from kavalai.agents.run_context import RunContext
from kavalai.agents.workflow_model import Task, TypeInputInfo


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
    info = TypeInputInfo(type="literal", value="hello")
    assert await rc.resolve_input_info(info) == "hello"


@pytest.mark.asyncio
async def test_resolve_input_info_context():
    rc = RunContext(data={"input": {"text": "world"}})

    # Using value as path
    info = TypeInputInfo(type="context", value="input.text")
    assert await rc.resolve_input_info(info) == "world"

    # Using name as path
    info = TypeInputInfo(type="context", name="input.text")
    assert await rc.resolve_input_info(info) == "world"

    # Path is None
    info = TypeInputInfo(type="context")
    assert await rc.resolve_input_info(info) is None


@pytest.mark.asyncio
async def test_resolve_input_info_history():
    session_id = uuid4()
    mock_service = AsyncMock()
    mock_service.get_history_value.return_value = "history_val"

    rc = RunContext(session_id=session_id, agent_service=mock_service)

    # Using value
    info = TypeInputInfo(type="history", value="some_key")
    assert await rc.resolve_input_info(info) == "history_val"
    mock_service.get_history_value.assert_called_with(session_id, "some_key")

    # Using name
    info = TypeInputInfo(type="history", name="other_key")
    assert await rc.resolve_input_info(info) == "history_val"
    mock_service.get_history_value.assert_called_with(session_id, "other_key")


@pytest.mark.asyncio
async def test_resolve_input_info_history_missing_service(caplog):
    rc = RunContext(session_id=uuid4())
    info = TypeInputInfo(type="history", value="key")
    assert await rc.resolve_input_info(info) is None
    assert (
        "Cannot load from history: agent_service or session_id not set" in caplog.text
    )


@pytest.mark.asyncio
async def test_prepare_tool_inputs():
    rc = RunContext(data={"val": "from_context"})
    task = Task(
        name="test_task",
        inputs={
            "lit": TypeInputInfo(type="literal", value="fixed"),
            "ctx": TypeInputInfo(type="context", value="val"),
            "implicit": TypeInputInfo(type="context"),  # should use name "implicit"
        },
    )
    # Patch implicit context to match data
    rc.data["implicit"] = "implicit_val"

    inputs = await rc.prepare_tool_inputs(task)
    assert inputs == {"lit": "fixed", "ctx": "from_context", "implicit": "implicit_val"}


@pytest.mark.asyncio
async def test_prepare_tool_inputs_with_pydantic_model():
    rc = RunContext()
    model_instance = MockModel(field="test")
    task = Task(
        name="test_task",
        inputs={"mod": TypeInputInfo(type="literal", value=model_instance)},
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
async def test_evaluate_condition_unknown_key():
    rc = RunContext()
    # If the key is not an operator, 'all', 'any', or 'not', it should fall through to return True (line 135)
    assert await rc.evaluate_condition({"unknown": "value"}) is True


@pytest.mark.asyncio
async def test_evaluate_condition_empty():
    rc = RunContext()
    assert await rc.evaluate_condition({}) is True
    assert await rc.evaluate_condition(None) is True
