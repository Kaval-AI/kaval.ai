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
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kavalai.agents.workflow_model import WorkflowModel, TypeInputInfo

logger = logging.getLogger(__name__)


def validate_rest_server_env_vars(workflow_model: "WorkflowModel"):
    """Validate that environment variables for REST server auth are defined."""
    from kavalai.agents.workflow_model import WorkflowException

    for server in workflow_model.rest_servers:
        # 1. URL Configuration validation
        if server.url and server.url_env:
            raise WorkflowException(
                f"REST server '{server.name}': Only one of 'url' or 'url_env' can be specified."
            )
        if not server.url and not server.url_env:
            raise WorkflowException(
                f"REST server '{server.name}': Either 'url' or 'url_env' must be specified."
            )

        # 2. URL Resolution check from environment (no mutation)
        if server.url_env:
            if server.url_env not in os.environ:
                raise WorkflowException(
                    f"Environment variable '{server.url_env}' for REST server "
                    f"'{server.name}' URL is not defined."
                )
            url_val = os.environ[server.url_env]
            # Check format of the env var value
            if not (url_val.startswith("http://") or url_val.startswith("https://")):
                raise WorkflowException(
                    f"REST server '{server.name}' has an invalid URL from {server.url_env}: {url_val}. "
                    f"It must start with http:// or https://"
                )

        # 3. URL Format validation for static URL
        if server.url and not (
            server.url.startswith("http://") or server.url.startswith("https://")
        ):
            raise WorkflowException(
                f"REST server '{server.name}' has an invalid URL: {server.url}. "
                f"It must start with http:// or https://"
            )

        # 4. Auth validation
        if server.username_env and server.password_env:
            if server.username_env not in os.environ:
                raise WorkflowException(
                    f"Environment variable '{server.username_env}' for REST server "
                    f"'{server.name}' username is not defined."
                )
            if server.password_env not in os.environ:
                raise WorkflowException(
                    f"Environment variable '{server.password_env}' for REST server "
                    f"'{server.name}' password is not defined."
                )
        elif server.username_env or server.password_env:
            # Only one of them is defined - this is an error
            raise WorkflowException(
                f"REST server '{server.name}' must have both username_env and "
                f"password_env defined, or neither."
            )

    for server in workflow_model.mcp_servers:
        if server.command_env:
            if server.command_env not in os.environ:
                raise WorkflowException(
                    f"Environment variable '{server.command_env}' for MCP server "
                    f"'{server.name}' command is not defined."
                )
        elif server.url_env:
            if server.url_env not in os.environ:
                raise WorkflowException(
                    f"Environment variable '{server.url_env}' for MCP server "
                    f"'{server.name}' URL is not defined."
                )
        elif not server.command and not server.url:
            raise WorkflowException(
                f"MCP server '{server.name}' must have either 'command', 'command_env', 'url', or 'url_env'."
            )


def get_root_context_name(info: "TypeInputInfo", fallback: str) -> str:
    """Extract the root context name from a TypeInputInfo (e.g., 'input' from 'input.user_message')."""
    path = info.value or info.name or fallback
    return str(path).split(".")[0]


def validate_workflow(workflow_model: "WorkflowModel"):
    from kavalai.agents.workflow_model import WorkflowException

    validate_rest_server_env_vars(workflow_model)

    if not (0.0 <= workflow_model.temperature <= 2.0):
        raise WorkflowException(
            f"Workflow temperature must be between 0.0 and 2.0, got {workflow_model.temperature}"
        )

    available_data = {"input"}
    for task in workflow_model.tasks:
        if task.temperature is not None and not (0.0 <= task.temperature <= 2.0):
            raise WorkflowException(
                f"Task '{task.name}' temperature must be between 0.0 and 2.0, got {task.temperature}"
            )

        # Check outputs
        if isinstance(task.output, str) and task.output:
            if task.output not in workflow_model.data_types:
                raise WorkflowException(
                    f"output '{task.output}' in task '{task.name}' is not defined in data_types."
                )
        elif isinstance(task.output, dict):
            # For combine task with dict output, we expect it to produce 'output' data type
            if "output" not in workflow_model.data_types:
                raise WorkflowException(
                    f"Task '{task.name}' has dict output but 'output' data type is not defined."
                )

        # Check inputs
        for input_name, input_info in task.inputs.items():
            if input_info.type == "context":
                root_name = get_root_context_name(input_info, input_name)
                if root_name not in available_data:
                    raise WorkflowException(
                        f"input '{root_name}' in task '{task.name}' is not available. "
                        f"Available context: {sorted(list(available_data))}"
                    )

        # Check MCP server existence
        if task.mcp_server:
            mcp_server_names = [s.name for s in workflow_model.mcp_servers]
            if task.mcp_server not in mcp_server_names:
                raise WorkflowException(
                    f"Task '{task.name}' refers to undefined MCP server '{task.mcp_server}'."
                )

        # Check allowed_mcp_servers existence
        if task.allowed_mcp_servers:
            mcp_server_names = [s.name for s in workflow_model.mcp_servers]
            for s_name in task.allowed_mcp_servers:
                if s_name not in mcp_server_names:
                    raise WorkflowException(
                        f"Task '{task.name}' refers to undefined allowed MCP server '{s_name}'."
                    )

        # After task success, its output is available for next tasks
        if isinstance(task.output, str) and task.output:
            available_data.add(task.output)
