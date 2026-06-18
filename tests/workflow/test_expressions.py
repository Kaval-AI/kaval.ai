import pytest

from kavalai.workflow.expressions import (
    ExpressionError,
    evaluate_bool,
    evaluate_expression,
    evaluate_value,
)


@pytest.fixture
def context():
    return {
        "state": {"count": 5, "status": "ok", "ready": True, "name": None},
        "input": {"user_message": "hello"},
        "items": [{"title": "a"}, {"title": "b"}],
        "tags": ["x", "y"],
    }


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("state.count > 3", True),
        ("state.count >= 5", True),
        ("state.count < 3", False),
        ("state.count <= 5", True),
        ("state.count == 5", True),
        ("state.count != 5", False),
        ("state.status == 'ok'", True),
        ("state.status == 'err'", False),
        ("state.count > 3 and state.status == 'ok'", True),
        ("state.count > 30 or state.status == 'ok'", True),
        ("not state.ready", False),
        ("state.ready", True),
        ("'x' in tags", True),
        ("'z' in tags", False),
        ("'z' not in tags", True),
        ("state.name is None", True),
        ("state.status is not None", True),
        ("state.count + 1 == 6", True),
        ("state.count - 5 == 0", True),
        ("state.count * 2 == 10", True),
        ("state.count / 2 == 2.5", True),
        ("state.count // 2 == 2", True),
        ("state.count % 2 == 1", True),
        ("-state.count == -5", True),
        ("+state.count == 5", True),
        ("items[0].title == 'a'", True),
        ("items[1].title == 'b'", True),
        ("input.user_message == 'hello'", True),
        ("1 < state.count < 10", True),
        ("1 < state.count < 3", False),
    ],
)
def test_expressions(context, expr, expected):
    assert evaluate_expression(expr, context) == expected


def test_subscript_access(context):
    assert evaluate_expression("tags[0]", context) == "x"
    assert evaluate_expression("state['status']", context) == "ok"


def test_subscript_out_of_range_returns_none(context):
    assert evaluate_expression("tags[10]", context) is None
    assert evaluate_expression("state['missing']", context) is None


def test_subscript_on_none_returns_none(context):
    assert evaluate_expression("state.name[0]", context) is None


def test_list_tuple_dict_literals(context):
    assert evaluate_expression("[1, 2, 3]", context) == [1, 2, 3]
    assert evaluate_expression("(1, 2)", context) == (1, 2)
    assert evaluate_expression("{'a': 1}", context) == {"a": 1}


def test_unknown_name_resolves_to_none(context):
    assert evaluate_expression("missing", context) is None
    assert evaluate_expression("missing is None", context) is True
    assert evaluate_expression("state.nope", context) is None


def test_boolop_returns_operand_values(context):
    # 'and' returns the first falsy operand (None here), not a bare bool.
    assert evaluate_expression("state.ready and missing", context) is None
    # 'or' returns the first truthy operand.
    assert evaluate_expression("missing or state.status", context) == "ok"
    # 'or' with all-falsy returns the last operand.
    assert evaluate_expression("missing or other_missing", context) is None


def test_invert_operator_rejected(context):
    with pytest.raises(ExpressionError):
        evaluate_expression("~state.count", context)


def test_evaluate_bool(context):
    assert evaluate_bool("state.count > 3", context) is True
    assert evaluate_bool("state.count > 30", context) is False
    assert evaluate_bool("missing", context) is False


def test_evaluate_value(context):
    assert evaluate_value("state.status", context) == "ok"
    assert evaluate_value("state.count", context) == "5"
    assert evaluate_value("state.ready", context) == "true"
    assert evaluate_value("state.count > 30", context) == "false"


def test_empty_expression_raises():
    with pytest.raises(ExpressionError):
        evaluate_expression("   ", {})
    with pytest.raises(ExpressionError):
        evaluate_expression(None, {})


def test_syntax_error_raises():
    with pytest.raises(ExpressionError):
        evaluate_expression("state.count >", {})


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os')",
        "open('x')",
        "len(tags)",
        "lambda: 1",
        "[x for x in tags]",
        "state.count ** 2",
        "state.count & 1",
    ],
)
def test_disallowed_constructs_raise(context, expr):
    with pytest.raises(ExpressionError):
        evaluate_expression(expr, context)


def test_type_error_wrapped(context):
    # Comparing None > int raises TypeError, surfaced as ExpressionError.
    with pytest.raises(ExpressionError):
        evaluate_expression("state.name > 3", context)
