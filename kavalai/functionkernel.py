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

import importlib
import inspect
import json
import logging
import os
from typing import Any, Dict, List, Optional, Callable

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from pydantic import BaseModel

from kavalai.agents.workflow_model import (
    McpServer,
    RestServer,
    WorkflowException,
)

logger = logging.getLogger(__name__)


class FunctionKernel:
    """
    Manages registration and execution of tools (REST, MCP, Python).
    Handles conversions and MCP session management.
    """

    def __init__(self):
        self.rest_servers: Dict[str, RestServer] = {}
        self.mcp_servers: Dict[str, McpServer] = {}
        self.python_tools: Dict[str, Callable] = {}

        # MCP session management
        self.mcp_sessions: Dict[str, ClientSession] = {}
        self.mcp_cleanups: List[Any] = []

    def register_rest_server(self, server: RestServer):
        if server.name in self.rest_servers:
            raise WorkflowException(
                f"REST server '{server.name}' is already registered."
            )
        self.rest_servers[server.name] = server

    def register_mcp_server(self, server: McpServer):
        if server.name in self.mcp_servers:
            raise WorkflowException(
                f"MCP server '{server.name}' is already registered."
            )
        self.mcp_servers[server.name] = server

    def register_python_tool(self, name: str, func: Callable):
        if name in self.python_tools:
            raise WorkflowException(f"Python tool '{name}' is already registered.")
        self.python_tools[name] = func

    async def call_tool(
        self,
        tool_uri: str,
        arguments: Dict[str, Any],
        output_type: Optional[type] = None,
        **kwargs,
    ) -> Any:
        """
        Unified tool call interface.
        Format: protocol://[name|module].function_name
        Example: python://kavalai.mytool.myfunc or rest://myrestserver.restfunction
        """
        if "://" not in tool_uri:
            raise WorkflowException(
                f"Invalid tool URI format: '{tool_uri}'. Expected protocol://[name|module].function_name"
            )

        protocol, path = tool_uri.split("://", 1)
        if "." not in path:
            raise WorkflowException(
                f"Invalid tool path format: '{path}'. Expected [name|module].function_name"
            )

        name_or_module, function_name = path.rsplit(".", 1)

        if protocol == "python":
            return await self.call_python_tool(
                f"{name_or_module}.{function_name}", arguments, output_type
            )
        elif protocol == "rest":
            method = kwargs.get("method", "get")
            return await self.call_rest_tool(
                name_or_module, function_name, arguments, method, output_type
            )
        elif protocol == "mcp":
            return await self.call_mcp_tool(
                name_or_module, function_name, arguments, output_type
            )
        else:
            raise WorkflowException(f"Unsupported protocol: '{protocol}'")

    async def close(self):
        """Cleanup all MCP sessions."""
        for session in reversed(self.mcp_cleanups):
            try:
                # Some are async context managers, some are ClientSessions
                if hasattr(session, "__aexit__"):
                    await session.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error during MCP cleanup: {e}")
        self.mcp_sessions.clear()
        self.mcp_cleanups.clear()

    async def call_rest_tool(
        self,
        server_name: str,
        tool: str,
        arguments: Dict[str, Any],
        method: str = "get",
        output_type: Optional[type] = None,
    ) -> Any:
        if server_name not in self.rest_servers:
            raise WorkflowException(f"REST server '{server_name}' not registered.")

        server = self.rest_servers[server_name]
        url = server.url
        if not url and server.url_env:
            url = os.environ.get(server.url_env)

        if not url:
            raise WorkflowException(f"URL for REST server '{server_name}' not found.")

        auth = None
        if server.username_env and server.password_env:
            username = os.environ.get(server.username_env)
            password = os.environ.get(server.password_env)
            if username and password:
                auth = (username, password)

        async with httpx.AsyncClient(auth=auth) as client:
            kwargs = {
                "params": arguments,
                "timeout": 60.0,
            }
            if method.lower() in ("post", "put", "patch"):
                kwargs["json"] = arguments
                kwargs.pop("params", None)

            full_url = f"{url.rstrip('/')}/{tool.lstrip('/')}"
            logger.info(f"Calling {method.upper()} {full_url}")
            response = await client.request(method.upper(), full_url, **kwargs)
            response.raise_for_status()
            result_data = response.json()

        if (
            output_type
            and issubclass(output_type, BaseModel)
            and isinstance(result_data, dict)
        ):
            return output_type(**result_data)
        return result_data

    async def _get_mcp_session(self, server_name: str) -> ClientSession:
        if server_name in self.mcp_sessions:
            return self.mcp_sessions[server_name]

        if server_name not in self.mcp_servers:
            raise WorkflowException(f"MCP server '{server_name}' not registered.")

        config = self.mcp_servers[server_name]

        if config.url or config.url_env:
            url = config.url
            if not url and config.url_env:
                url = os.environ.get(config.url_env)

            if not url:
                raise WorkflowException(
                    f"URL for MCP server '{server_name}' not found."
                )

            logger.info(f"Connecting to HTTP MCP server {server_name} at {url}")
            aclient = sse_client(url)
            read, write = await aclient.__aenter__()
            self.mcp_cleanups.append(aclient)
        else:
            command = config.command
            if not command and config.command_env:
                command = os.environ.get(config.command_env)

            if not command:
                raise WorkflowException(
                    f"Command for MCP server '{server_name}' not found."
                )

            server_params = StdioServerParameters(
                command=command,
                args=config.args,
                env={**os.environ, **config.env},
            )
            logger.info(f"Connecting to stdio MCP server {server_name}")
            aclient = stdio_client(server_params)
            read, write = await aclient.__aenter__()
            self.mcp_cleanups.append(aclient)

        session = ClientSession(read, write)
        await session.__aenter__()
        self.mcp_cleanups.append(session)
        await session.initialize()
        self.mcp_sessions[server_name] = session
        return session

    async def call_mcp_tool(
        self,
        server_name: str,
        tool: str,
        arguments: Dict[str, Any],
        output_type: Optional[type] = None,
    ) -> Any:
        session = await self._get_mcp_session(server_name)

        logger.info(f"Calling MCP tool {server_name}/{tool}")
        response = await session.call_tool(tool, arguments=arguments)

        if response.isError:
            raise WorkflowException(
                f"MCP tool '{tool}' on server '{server_name}' failed: {response.content}"
            )

        result_data = None
        for content in response.content:
            if hasattr(content, "text"):
                try:
                    result_data = json.loads(content.text)
                except (json.JSONDecodeError, TypeError):
                    result_data = content.text
                break

        if (
            output_type
            and issubclass(output_type, BaseModel)
            and isinstance(result_data, dict)
        ):
            return output_type(**result_data)
        return result_data

    async def call_python_tool(
        self,
        python_tool: str,
        arguments: Dict[str, Any],
        output_type: Optional[type] = None,
    ) -> Any:
        if python_tool in self.python_tools:
            func = self.python_tools[python_tool]
        else:
            try:
                module_name, func_name = python_tool.rsplit(".", 1)
                module = importlib.import_module(module_name)
                func = getattr(module, func_name)
            except (ValueError, ImportError, AttributeError) as e:
                raise WorkflowException(
                    f"Failed to load python_tool '{python_tool}': {e}"
                )

        sig = inspect.signature(func)
        try:
            bound_args = sig.bind(**arguments)
        except TypeError as e:
            raise WorkflowException(
                f"Python tool '{python_tool}' signature mismatch: {e}"
            )

        try:
            if inspect.iscoroutinefunction(func):
                result = await func(*bound_args.args, **bound_args.kwargs)
            else:
                result = func(*bound_args.args, **bound_args.kwargs)
        except Exception as e:
            logger.exception(f"Error executing python_tool '{python_tool}'")
            raise WorkflowException(f"Error executing python_tool '{python_tool}': {e}")

        if output_type and issubclass(output_type, BaseModel):
            try:
                if isinstance(result, dict):
                    return output_type(**result)
                elif isinstance(result, BaseModel):
                    return output_type(**result.model_dump())
                elif isinstance(result, output_type):
                    return result
                else:
                    fields = output_type.model_fields
                    if len(fields) == 1:
                        field_name = list(fields.keys())[0]
                        return output_type(**{field_name: result})
                    else:
                        return output_type(result)
            except Exception as e:
                raise WorkflowException(
                    f"Python tool '{python_tool}' returned incompatible result: {e}"
                )
        return result

    async def get_tool_descriptions(self) -> str:
        """Returns a string description of all registered tools for prompts."""
        descriptions = []

        # Python tools
        for name in self.python_tools.keys():
            descriptions.append(f"python://{name}")

        # REST tools - handled dynamically, but we could list registration prefix
        for name in self.rest_servers.keys():
            # For REST we don't have a fixed list of functions yet, but we use the name
            descriptions.append(f"rest://{name}.<function_name>")

        # MCP tools
        for name in self.mcp_servers.keys():
            try:
                session = await self._get_mcp_session(name)
                tools_result = await session.list_tools()
                for tool in tools_result.tools:
                    # Format: mcp://server_name.tool_name
                    descriptions.append(f"mcp://{name}.{tool.name}")
            except Exception as e:
                logger.warning(f"Could not list tools for MCP server {name}: {e}")

        return "\n".join(descriptions)
