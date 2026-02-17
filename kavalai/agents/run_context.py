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

import logging
import operator
from typing import Optional, Any
from uuid import UUID

from pydantic import BaseModel

from kavalai.agents.resolvers import resolve_path
from kavalai.agents.workflow_model import Task, TypeInputInfo

logger = logging.getLogger(__name__)


class RunContext(BaseModel):
    """Runtime data for a single interaction."""

    agent_id: Optional[UUID] = None
    session_id: Optional[UUID] = None
    run_id: Optional[UUID] = None
    data: dict = {}
    agent_service: Optional[Any] = None

    def resolve_context_value(self, path: str):
        """Resolve a dotted path like 'input.user_message' from context data."""
        return resolve_path(self.data, path)

    async def resolve_input_info(self, info: TypeInputInfo):
        """Resolve a TypeInputInfo to its actual value."""
        if info.type == "literal":
            return info.value
        if info.type == "history":
            if not self.agent_service or not self.session_id:
                logger.warning(
                    "Cannot load from history: agent_service or session_id not set"
                )
                return None
            path = info.value or info.name
            return await self.agent_service.get_history_value(
                self.session_id, str(path)
            )

        # For context type, use info.value (the path) or info.name
        path = info.value or info.name
        if path:
            return self.resolve_context_value(str(path))
        return None

    async def prepare_tool_inputs(self, task: Task) -> dict:
        inputs = {}
        for name, info in task.inputs.items():
            if info.value is None and info.name is None:
                info = info.model_copy(update={"value": name})
            value = await self.resolve_input_info(info)
            if isinstance(value, BaseModel):
                value = value.model_dump()
            inputs[name] = value

        return inputs

    async def evaluate_condition(self, condition: dict) -> bool:
        """
        Evaluate a condition dictionary.
        Supported formats:
        - { "eq": [val1, val2] }
        - { "all": [cond1, cond2] }
        - { "any": [cond1, cond2] }
        - { "not": cond }
        Values can be TypeInputInfo (dict) or raw literals.
        """
        if not condition:
            return True

        operators = {
            "eq": operator.eq,
            "not_eq": operator.ne,
            "gt": operator.gt,
            "gte": operator.ge,
            "lt": operator.lt,
            "lte": operator.le,
            "contains": lambda a, b: b in a if a is not None else False,
        }

        for key, val in condition.items():
            if key in operators:
                if not isinstance(val, list) or len(val) != 2:
                    raise ValueError(f"Operator '{key}' requires a list of 2 operands.")

                operands = []
                for operand in val:
                    if isinstance(operand, dict) and "type" in operand:
                        # It's a TypeInputInfo
                        info = TypeInputInfo(**operand)
                        operands.append(await self.resolve_input_info(info))
                    else:
                        operands.append(operand)

                return operators[key](operands[0], operands[1])

            elif key == "is_null":
                operand = val
                if isinstance(operand, dict) and "type" in operand:
                    info = TypeInputInfo(**operand)
                    operand = await self.resolve_input_info(info)
                return operand is None

            elif key == "is_not_null":
                operand = val
                if isinstance(operand, dict) and "type" in operand:
                    info = TypeInputInfo(**operand)
                    operand = await self.resolve_input_info(info)
                return operand is not None

            elif key == "all":
                if not isinstance(val, list):
                    raise ValueError("'all' requires a list of conditions.")
                results = []
                for c in val:
                    results.append(await self.evaluate_condition(c))
                return all(results)

            elif key == "any":
                if not isinstance(val, list):
                    raise ValueError("'any' requires a list of conditions.")
                results = []
                for c in val:
                    results.append(await self.evaluate_condition(c))
                return any(results)

            elif key == "not":
                if not isinstance(val, dict):
                    raise ValueError("'not' requires a single condition dictionary.")
                return not await self.evaluate_condition(val)

        return True
