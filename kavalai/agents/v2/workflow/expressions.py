"""
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import ast
import operator
from typing import Any

from kavalai.agents.resolvers import resolve_path


class ExpressionError(ValueError):
    """Raised when an expression cannot be parsed or safely evaluated."""


# Whitelisted binary, comparison, boolean and unary operators. Anything not in
# these maps (e.g. bitwise ops, power, matmul) is rejected.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

_COMPARE_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b if b is not None else False,
    ast.NotIn: lambda a, b: a not in b if b is not None else True,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}

_UNARY_OPS = {
    ast.Not: operator.not_,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def evaluate_expression(expr: str, context: dict) -> Any:
    """Safely evaluate a simple string expression against ``context``.

    Supports comparisons (``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``,
    ``in``, ``not in``, ``is``, ``is not``), boolean logic (``and``, ``or``,
    ``not``), arithmetic (``+``, ``-``, ``*``, ``/``, ``//``, ``%``), literals,
    and list/tuple/dict displays. Names and attribute/subscript chains
    (``state.count``, ``input.user_message``, ``items[0].title``) are resolved
    from ``context`` via :func:`resolve_path`; unknown names resolve to
    ``None``.

    Arbitrary code is rejected: function calls, lambdas, comprehensions,
    imports, attribute writes, etc. all raise :class:`ExpressionError`.
    """
    if not isinstance(expr, str) or not expr.strip():
        raise ExpressionError("Expression must be a non-empty string.")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"Could not parse expression '{expr}': {e}") from e
    try:
        return _eval_node(tree.body, context)
    except ExpressionError:
        raise
    except Exception as e:
        raise ExpressionError(f"Error evaluating expression '{expr}': {e}") from e


def evaluate_bool(expr: str, context: dict) -> bool:
    """Evaluate ``expr`` and coerce the result to a bool (for ``if`` nodes)."""
    return bool(evaluate_expression(expr, context))


def evaluate_value(expr: str, context: dict) -> str:
    """Evaluate ``expr`` and stringify the result (for ``switch`` case lookup)."""
    value = evaluate_expression(expr, context)
    if isinstance(value, bool):
        # Normalise booleans to lowercase to match YAML-authored case keys.
        return "true" if value else "false"
    return str(value)


def _eval_node(node: ast.AST, context: dict) -> Any:
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        return resolve_path(context, node.id)

    if isinstance(node, ast.Attribute):
        base = _eval_node(node.value, context)
        return resolve_path(base, node.attr)

    if isinstance(node, ast.Subscript):
        base = _eval_node(node.value, context)
        key = _eval_node(node.slice, context)
        if base is None:
            return None
        try:
            return base[key]
        except (KeyError, IndexError, TypeError):
            return None

    if isinstance(node, ast.BoolOp):
        values = node.values
        if isinstance(node.op, ast.And):
            result: Any = True
            for value in values:
                result = _eval_node(value, context)
                if not result:
                    return result
            return result
        # Or: return the first truthy operand, else the last.
        result = False
        for value in values:
            result = _eval_node(value, context)
            if result:
                return result
        return result

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ExpressionError(
                f"Unsupported unary operator: {type(node.op).__name__}"
            )
        return op(_eval_node(node.operand, context))

    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ExpressionError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_eval_node(node.left, context), _eval_node(node.right, context))

    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, context)
        for op_node, comparator in zip(node.ops, node.comparators):
            op = _COMPARE_OPS.get(type(op_node))
            if op is None:  # pragma: no cover - all Python comparison ops are mapped
                raise ExpressionError(
                    f"Unsupported comparison: {type(op_node).__name__}"
                )
            right = _eval_node(comparator, context)
            if not op(left, right):
                return False
            left = right
        return True

    if isinstance(node, ast.List):
        return [_eval_node(elt, context) for elt in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(elt, context) for elt in node.elts)

    if isinstance(node, ast.Dict):
        return {
            _eval_node(k, context): _eval_node(v, context)
            for k, v in zip(node.keys, node.values)
        }

    raise ExpressionError(f"Unsupported expression element: {type(node).__name__}")
