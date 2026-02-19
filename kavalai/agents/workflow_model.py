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

from typing import Optional, Literal
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, model_validator


def to_plain(obj):
    """Recursively convert Pydantic models, dicts, and lists into plain JSON-serializable types."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, (datetime, UUID)):
        return str(obj)
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj


class WorkflowException(Exception):
    pass


class TypeInputInfo(BaseModel):
    type: Literal["literal", "context", "history"]
    value: Optional[BaseModel | str | int | float | bool] = None
    name: Optional[str] = None


class Task(BaseModel):
    name: str
    inputs: dict[str, TypeInputInfo] = {}
    output: str | dict[str, TypeInputInfo] = ""
    when: Optional[dict] = None
    stop: bool = False

    @model_validator(mode="after")
    def validate_conditions(self) -> "Task":
        if self.when:
            self._validate_condition(self.when)
        return self

    def _validate_condition(self, condition: dict):
        operators = {"eq", "not_eq", "gt", "gte", "lt", "lte", "contains", "len"}
        for key, val in condition.items():
            if key in operators:
                if not isinstance(val, list) or len(val) != 2:
                    raise ValueError(f"Operator '{key}' requires a list of 2 operands.")
            elif key in ["is_null", "is_not_null", "is_true"]:
                # These take a single operand (either dict or literal)
                pass
            elif key == "all":
                if not isinstance(val, list):
                    raise ValueError("'all' requires a list of conditions.")
                for c in val:
                    self._validate_condition(c)
            elif key == "any":
                if not isinstance(val, list):
                    raise ValueError("'any' requires a list of conditions.")
                for c in val:
                    self._validate_condition(c)
            elif key == "not":
                if not isinstance(val, dict):
                    raise ValueError("'not' requires a single condition dictionary.")
                self._validate_condition(val)

    # LLM call
    prompt: Optional[str] = None
    temperature: Optional[float] = None
    use_history: bool = True
    # REST tool call
    tool: Optional[str] = None
    rest_server: Optional[str] = None
    method: str = "get"
    # LLM images
    images: list[TypeInputInfo] = []
    # Streaming
    stream: bool = False
    # MCP tool call
    mcp_server: Optional[str] = None


class RestServer(BaseModel):
    name: str
    url: Optional[str] = None
    url_env: Optional[str] = None
    username_env: Optional[str] = None
    password_env: Optional[str] = None

    @model_validator(mode="after")
    def check_url_configs(self) -> "RestServer":
        if self.url and self.url_env:
            raise ValueError(
                f"REST server '{self.name}': Only one of 'url' or 'url_env' can be specified."
            )
        if not self.url and not self.url_env:
            raise ValueError(
                f"REST server '{self.name}': Either 'url' or 'url_env' must be specified."
            )
        return self


class McpServer(BaseModel):
    name: str
    command: Optional[str] = None
    command_env: Optional[str] = None
    args: list[str] = []
    env: dict[str, str] = {}

    @model_validator(mode="after")
    def check_command_configs(self) -> "McpServer":
        if self.command and self.command_env:
            raise ValueError(
                f"MCP server '{self.name}': Only one of 'command' or 'command_env' can be specified."
            )
        if not self.command and not self.command_env:
            raise ValueError(
                f"MCP server '{self.name}': Either 'command' or 'command_env' must be specified."
            )
        return self


class WorkflowModel(BaseModel):
    name: str
    description: str = ""
    version: str = "1.0"
    temperature: float = 0.0
    llm_model: Optional[str] = None
    data_types: dict[str, dict]
    rest_servers: list[RestServer] = []
    mcp_servers: list[McpServer] = []
    tasks: list[Task]


class WorkflowRunResult(BaseModel):
    session_id: Optional[UUID] = None
    data: Optional[BaseModel] = None
    run_context: Optional[BaseModel] = None
