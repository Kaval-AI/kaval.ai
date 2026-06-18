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

Shared workflow building blocks reused across the codebase (input wiring,
server/tool declarations and the workflow exception). The v2 workflow engine
lives in ``kavalai.workflow``.
"""

from typing import Optional, Literal

from pydantic import BaseModel, model_validator


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
