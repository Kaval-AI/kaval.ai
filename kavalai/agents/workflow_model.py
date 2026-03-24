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

from typing import Optional, Literal, Union, Annotated, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, model_validator, Field, PrivateAttr, ConfigDict

from .utils import clean_text


def to_plain(obj):
    """Recursively convert Pydantic models, dicts, and lists into plain JSON-serializable types."""
    if isinstance(obj, str):
        return clean_text(obj)
    if isinstance(obj, BaseModel):
        return to_plain(obj.model_dump())
    if isinstance(obj, (datetime, UUID)):
        return str(obj)
    if isinstance(obj, dict):
        # Filter out internal attributes that might not be serializable or cause issues
        res = {}
        for k, v in obj.items():
            try:
                sk = clean_text(str(k))
                if sk.startswith("_") or sk == "metadata":
                    continue
                res[sk] = to_plain(v)
            except Exception:
                continue
        return res
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj


class WorkflowException(Exception):
    pass


class YamlModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    _line_number: Optional[int] = PrivateAttr(default=None)
    _file_path: Optional[str] = PrivateAttr(default=None)

    def __init__(self, **data: Any):
        line = data.pop("__line__", None)
        file_path = data.pop("__file_path__", None)
        super().__init__(**data)
        self._line_number = line
        self._file_path = file_path

    model_config = ConfigDict(extra="ignore")

    @property
    def line_number(self) -> Optional[int]:
        return self._line_number

    @property
    def file_path(self) -> Optional[str]:
        return self._file_path


class TypeInputInfo(YamlModel):
    type: Literal["literal", "context", "history"]
    value: Optional[BaseModel | str | int | float | bool] = None
    name: Optional[str] = None


class BaseTask(YamlModel):
    name: str
    inputs: dict[str, TypeInputInfo] = {}
    output: str | dict[str, TypeInputInfo] = ""
    when: Optional[dict] = None
    stop: bool = False
    stream_updates: bool = False
    stream_output: bool = False

    @model_validator(mode="before")
    @classmethod
    def check_deprecated_stream(cls, data: Any) -> Any:
        if isinstance(data, dict) and "stream" in data:
            raise ValueError(
                "The 'stream' field is deprecated and no longer supported. "
                "Please use 'stream_updates' and/or 'stream_output' instead."
            )
        return data

    @model_validator(mode="after")
    def validate_conditions(self) -> "BaseTask":
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


class LLMTask(BaseTask):
    type: Literal["llm"] = "llm"
    prompt: str
    temperature: Optional[float] = None
    use_history: bool = True
    llm_model: Optional[str] = None
    llm_kwargs: dict[str, Any] = Field(default_factory=dict)


class RestTask(BaseTask):
    type: Literal["rest"] = "rest"
    tool: str
    rest_server: str
    method: str = "get"


class McpTask(BaseTask):
    type: Literal["mcp"] = "mcp"
    tool: str
    mcp_server: str


class PythonTask(BaseTask):
    type: Literal["python"] = "python"
    python_tool: str


class AgentTask(BaseTask):
    type: Literal["agent"] = "agent"
    max_steps: int = 1
    allowed_tools: list[str] = Field(default_factory=list)
    timeout: Optional[int] = None
    prompt: Optional[str] = None
    temperature: Optional[float] = None
    use_history: bool = False
    stream_persisted: bool = False
    llm_model: Optional[str] = None
    llm_kwargs: dict[str, Any] = Field(default_factory=dict)


class CombineTask(BaseTask):
    type: Literal["combine"] = "combine"


class RagQueryTask(BaseTask):
    type: Literal["rag_query"] = "rag_query"
    text: str
    top_k: int = 5
    collection_name: Optional[str] = None
    source_ids: Optional[list[str]] = None
    keep_best: bool = False


Task = Annotated[
    Union[LLMTask, RestTask, McpTask, PythonTask, AgentTask, CombineTask, RagQueryTask],
    Field(discriminator="type"),
]


class RestServer(YamlModel):
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


class McpServer(YamlModel):
    name: str
    command: Optional[str] = None
    command_env: Optional[str] = None
    args: list[str] = []
    env: dict[str, str] = {}
    url: Optional[str] = None
    url_env: Optional[str] = None

    @model_validator(mode="after")
    def check_configs(self) -> "McpServer":
        stdio_configured = bool(self.command or self.command_env)
        http_configured = bool(self.url or self.url_env)

        if stdio_configured and http_configured:
            raise ValueError(
                f"MCP server '{self.name}': Cannot specify both stdio (command/command_env) and HTTP (url/url_env) configurations."
            )
        if not stdio_configured and not http_configured:
            raise ValueError(
                f"MCP server '{self.name}': Either stdio (command/command_env) or HTTP (url/url_env) must be specified."
            )

        if self.command and self.command_env:
            raise ValueError(
                f"MCP server '{self.name}': Only one of 'command' or 'command_env' can be specified for stdio."
            )
        if self.url and self.url_env:
            raise ValueError(
                f"MCP server '{self.name}': Only one of 'url' or 'url_env' can be specified for HTTP."
            )
        return self


class PythonFunction(YamlModel):
    name: str
    path: str


class TemplateModel(YamlModel):
    name: str
    value: str


class WorkflowModel(YamlModel):
    name: str
    description: str = ""
    version: str = "1.0"
    temperature: float = 0.0
    llm_model: Optional[str] = None
    llm_kwargs: dict[str, Any] = Field(default_factory=dict)
    embedding_model: str = "openai/text-embedding-3-small"
    data_types: dict[str, dict]
    rest_servers: list[RestServer] = []
    mcp_servers: list[McpServer] = []
    templates: list[TemplateModel] = []
    python_functions: list[PythonFunction] = []
    tasks: list[Task]


class WorkflowRunResult(BaseModel):
    session_id: Optional[UUID] = None
    data: Optional[BaseModel] = None
    run_context: Optional[BaseModel] = None
