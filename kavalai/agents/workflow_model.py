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

from pydantic import BaseModel, model_validator, Field


class WorkflowException(Exception):
    pass


class ArgumentInfo(BaseModel):
    """Describes input arguments in workflow YAML files.

    The 'type' field describes where the input argument should be retrieved from.
    'literal' - use value as specified
    'context' - retrieve from agent run context
    'history' - retrieve from previous agent run contexts.

    """

    type: Literal["literal", "context", "history"]
    value: Optional[BaseModel | str | int | float | bool] = None
    name: Optional[str] = None


class BaseTask(BaseModel):
    """Common task functionality.

    Parameters
    ==========
    name: str - The name of the task.
    inputs: dict[str, ArgumentInfo] - A dictionary describing where to load
        the input arguments from.
    output: str - the name of the output variable.
    stop: bool - whether to stop the workflow after this task.
    stream_updates: bool - whether to send agent status updates to stream.
    stream_output: bool - whether to send the output of this task to stream.
    """

    name: str
    inputs: dict[str, ArgumentInfo] = {}
    output: str
    when: Optional[dict] = None
    stop: bool = False
    stream_updates: bool = False
    stream_output: bool = False

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
        return self


class BaseLLMTask(BaseTask):
    """Common properties of LLM and Agent tasks."""

    prompt: str
    use_history: bool = True
    llm_model: Optional[str] = None
    llm_kwargs: dict[str, Any] = Field(default_factory=dict)


class LLMTask(BaseLLMTask):
    type: Literal["llm"] = "llm"


class AgentTask(BaseLLMTask):
    """AgentTask defines a multi-step agent capable of using various tools.

    allowed_tools: list[str] - A list of allowed tools to use in the agent.
    max_steps: int - Defines the maximum amount of steps to run the agent.
    """

    type: Literal["agent"] = "agent"
    allowed_tools: list[str] = Field(default_factory=list)
    max_steps: int = 1
    timeout: Optional[int] = None
    stream_persisted: bool = False


class RestTask(BaseTask):
    """Defines a REST tool call."""

    type: Literal["rest"] = "rest"
    tool: str
    rest_server: str
    method: str = "get"


class McpTask(BaseTask):
    """Defines an MCP tool call task."""

    type: Literal["mcp"] = "mcp"
    tool: str
    mcp_server: str


class PythonTask(BaseTask):
    """Defines a Python tool call task."""

    type: Literal["python"] = "python"
    python_tool: str


class AssignTask(BaseTask):
    """Assign task creates new context variables out of existing ones."""

    type: Literal["combine"] = "combine"


class RagQueryTask(BaseTask):
    """Query RAG and store the results in context.
    TODO: refactor this under system tools.
    """

    type: Literal["rag_query"] = "rag_query"
    text: str
    top_k: int = 5
    collection_name: Optional[str] = None
    source_ids: Optional[list[str]] = None
    keep_best: bool = False


# Define combined Task data type using type field as a discriminator.
Task = Annotated[
    Union[LLMTask, RestTask, McpTask, PythonTask, AgentTask, AssignTask, RagQueryTask],
    Field(discriminator="type"),
]


class RestServer(BaseModel):
    """Defines a REST server.

    We also support HTTP Basic Auth for REST server endpoints, which are defined via
    environment variables username_env and password_env.

    Note that url_env can also be read from the env file.
    """

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
    """Defines an MCP server."""

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


class PythonFunction(BaseModel):
    name: str
    path: str


class TemplateModel(BaseModel):
    name: str
    value: str


class WorkflowModel(BaseModel):
    """Defines a workflow model.

    name: str - The name of the workflow, use agent name.
    description: str - A description of the workflow.
    version: str - The version of the workflow.
    llm_model: str - Default LLM model for the workflow. Task can override this.
    llm_kwargs: dict[str, Any] - Default LLM kwargs passed to tasks. Tasks can override this.
    embedding_model: str - Default embedding model for the workflow.
    rest_servers: list[RestServer] - List of REST servers.
    mcp_servers: list[McpServer] - List of MCP servers.
    python_functions: list[PythonFunction] - List of Python functions.
    templates: list[TemplateModel] - List of templates that can be used to build prompts from smaller pieces.
    tasks: list[Task] - List of tasks.
    """

    name: str
    description: str = ""
    version: str = "1.0"
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
