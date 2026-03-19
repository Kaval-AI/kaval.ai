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

import inspect
import json
from loguru import logger
import os
from typing import Any, Dict, List, Optional, Callable, Type

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, create_model

from kavalai.agents.workflow_model import (
    McpServer,
    RestServer,
    WorkflowException,
)

SSE_CLIENT_TIMEOUT_SECONDS = 30.0


class FunctionKernelException(WorkflowException):
    pass


def pythontool(func: Callable) -> Callable:
    """Decorator to mark a function as a kavalai tool."""
    func._is_kavalai_tool = True
    return func


class ToolDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    input_model: Type[BaseModel]
    output_model: Type[BaseModel]


class FunctionKernel:
    """
    Manages registration and execution of tools (REST, MCP, Python).
    Handles conversions and MCP session management.
    """

    def __init__(self):
        self.rest_servers: Dict[str, RestServer] = {}
        self.rest_tool_definitions: Dict[str, Dict[str, ToolDefinition]] = {}
        self.mcp_servers: Dict[str, McpServer] = {}
        self.python_tools: Dict[str, Callable] = {}
        self.python_tool_definitions: Dict[str, ToolDefinition] = {}
        self.mcp_tool_definitions: Dict[str, Dict[str, ToolDefinition]] = {}

        # MCP session management
        self.mcp_sessions: Dict[str, ClientSession] = {}
        self.mcp_cleanups: List[Any] = []

    def register_rest_server(self, server: RestServer):
        if server.name in self.rest_servers:
            raise FunctionKernelException(
                f"REST server '{server.name}' is already registered."
            )
        self.rest_servers[server.name] = server

    def register_rest_tool(
        self,
        server_name: str,
        tool_name: str,
        method: str,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        description: Optional[str] = None,
    ):
        if server_name not in self.rest_tool_definitions:
            self.rest_tool_definitions[server_name] = {}

        InputModel = self._create_model_from_jsonschema(
            f"{server_name}_{tool_name}_input", input_schema
        )
        OutputModel = self._create_model_from_jsonschema(
            f"{server_name}_{tool_name}_output", output_schema
        )

        self.rest_tool_definitions[server_name][tool_name] = ToolDefinition(
            name=tool_name,
            description=description,
            input_model=InputModel,
            output_model=OutputModel,
        )
        # Store method in a way it can be retrieved
        self.rest_tool_definitions[server_name][tool_name].description = json.dumps(
            {"method": method, "description": description}
        )

    def register_mcp_server(self, server: McpServer):
        if server.name in self.mcp_servers:
            raise FunctionKernelException(
                f"MCP server '{server.name}' is already registered."
            )
        self.mcp_servers[server.name] = server

    def register_python_tool(self, name: str, func: Callable):
        if not getattr(func, "_is_kavalai_tool", False):
            raise FunctionKernelException(
                f"Function '{func.__name__}' must be decorated with @kavalai.pythontool"
            )

        # Normalize name by removing protocol prefix if present
        if "://" in name:
            _, name = name.split("://", 1)

        if name in self.python_tools:
            raise FunctionKernelException(
                f"Python tool '{name}' is already registered."
            )
        self.python_tools[name] = func
        self.python_tool_definitions[name] = self._generate_python_tool_definition(
            name, func
        )

    def _generate_python_tool_definition(
        self, name: str, func: Callable
    ) -> ToolDefinition:
        sig = inspect.signature(func)

        # Input Model
        input_fields = {}
        for param_name, p in sig.parameters.items():
            annotation = (
                p.annotation if p.annotation != inspect.Parameter.empty else Any
            )
            default = p.default if p.default != inspect.Parameter.empty else ...
            input_fields[param_name] = (annotation, default)

        InputModel = create_model(f"{name}_input", **input_fields)

        # Output Model
        output_annotation = (
            sig.return_annotation
            if sig.return_annotation != inspect.Signature.empty
            else Any
        )
        if isinstance(output_annotation, type) and issubclass(
            output_annotation, BaseModel
        ):
            OutputModel = output_annotation
        else:
            OutputModel = create_model(
                f"{name}_output", result=(output_annotation, ...)
            )

        return ToolDefinition(
            name=name,
            description=inspect.getdoc(func) or "",
            input_model=InputModel,
            output_model=OutputModel,
        )

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
            raise FunctionKernelException(
                f"Invalid tool URI format: '{tool_uri}'. Expected protocol://[name|module].function_name"
            )

        protocol, path = tool_uri.split("://", 1)

        if protocol == "python":
            return await self._call_python_tool(path, arguments, output_type)

        if protocol == "rest" or protocol == "mcp":
            if "." not in path:
                raise FunctionKernelException(
                    f"Invalid tool path format: '{path}'. Expected [name|module].function_name"
                )
            name_or_module, function_name = path.rsplit(".", 1)

        if protocol == "rest":
            method = kwargs.get("method", "get")
            if (
                name_or_module in self.rest_tool_definitions
                and function_name in self.rest_tool_definitions[name_or_module]
            ):
                definition = self.rest_tool_definitions[name_or_module][function_name]
                try:
                    desc_data = json.loads(definition.description)
                    method = desc_data.get("method", method)
                except Exception:
                    pass

                # Validate input
                validated_args = definition.input_model(**arguments).model_dump()
                result = await self._call_rest_tool(
                    name_or_module,
                    function_name,
                    validated_args,
                    method,
                    output_type or definition.output_model,
                )
                return result

            return await self._call_rest_tool(
                name_or_module, function_name, arguments, method, output_type
            )
        elif protocol == "mcp":
            return await self._call_mcp_tool(
                name_or_module, function_name, arguments, output_type
            )
        else:
            raise FunctionKernelException(f"Unsupported protocol: '{protocol}'")

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

    async def _call_rest_tool(
        self,
        server_name: str,
        tool: str,
        arguments: Dict[str, Any],
        method: str = "get",
        output_type: Optional[type] = None,
    ) -> Any:
        if server_name not in self.rest_servers:
            raise FunctionKernelException(
                f"REST server '{server_name}' not registered."
            )

        server = self.rest_servers[server_name]
        url = server.url
        if not url and server.url_env:
            url = os.environ.get(server.url_env)

        if not url:
            raise FunctionKernelException(
                f"URL for REST server '{server_name}' not found."
            )

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
            raise FunctionKernelException(f"MCP server '{server_name}' not registered.")

        config = self.mcp_servers[server_name]

        if config.url or config.url_env:
            url = config.url
            if not url and config.url_env:
                url = os.environ.get(config.url_env)

            if not url:
                raise FunctionKernelException(
                    f"URL for MCP server '{server_name}' not found."
                )

            logger.info(f"Connecting to HTTP MCP server {server_name} at {url}")
            aclient = sse_client(url, timeout=SSE_CLIENT_TIMEOUT_SECONDS)
            read, write = await aclient.__aenter__()
            self.mcp_cleanups.append(aclient)
        else:
            command = config.command
            if not command and config.command_env:
                command = os.environ.get(config.command_env)

            if not command:
                raise FunctionKernelException(
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

        # Fetch and store tool definitions
        await self._refresh_mcp_tool_definitions(server_name, session)

        return session

    async def _refresh_mcp_tool_definitions(
        self, server_name: str, session: ClientSession
    ):
        try:
            tools_result = await session.list_tools()
            definitions = {}
            for tool in tools_result.tools:
                # MCP tool input schema is usually a JSON Schema
                # For now, we store the raw schema and we could dynamically create a Pydantic model
                # But to stay consistent with the "Pydantic models for everything" requirement:
                input_model = self._create_model_from_jsonschema(
                    f"{server_name}_{tool.name}_input", tool.inputSchema
                )
                # MCP doesn't strictly define output schema in tool list, so we use a generic one
                output_model = create_model(
                    f"{server_name}_{tool.name}_output", result=(Any, ...)
                )

                definitions[tool.name] = ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    input_model=input_model,
                    output_model=output_model,
                )
            self.mcp_tool_definitions[server_name] = definitions
        except Exception as e:
            logger.warning(f"Could not refresh tools for MCP server {server_name}: {e}")

    def _create_model_from_jsonschema(
        self, name: str, schema: Dict[str, Any]
    ) -> Type[BaseModel]:
        """Very basic JSON Schema to Pydantic model conversion."""
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        fields = {}

        type_map = {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
            "object": dict,
            "array": list,
        }

        for prop_name, prop_schema in properties.items():
            prop_type = prop_schema.get("type", "any")
            python_type = type_map.get(prop_type, Any)
            default = ... if prop_name in required else None
            fields[prop_name] = (python_type, default)

        return create_model(name, **fields)

    async def _call_mcp_tool(
        self,
        server_name: str,
        tool: str,
        arguments: Dict[str, Any],
        output_type: Optional[type] = None,
    ) -> Any:
        session = await self._get_mcp_session(server_name)

        # Validate arguments if definition exists
        if (
            server_name in self.mcp_tool_definitions
            and tool in self.mcp_tool_definitions[server_name]
        ):
            definition = self.mcp_tool_definitions[server_name][tool]
            try:
                validated_args = definition.input_model(**arguments).model_dump()
                arguments = validated_args
            except Exception as e:
                raise FunctionKernelException(
                    f"MCP tool '{tool}' on server '{server_name}' argument validation failed: {e}"
                )

        logger.info(f"Calling MCP tool {server_name}/{tool}")
        response = await session.call_tool(tool, arguments=arguments)

        if response.isError:
            raise FunctionKernelException(
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

        # Convert output using output_type or definition's output_model
        target_output_type = output_type
        if not target_output_type and server_name in self.mcp_tool_definitions:
            if tool in self.mcp_tool_definitions[server_name]:
                target_output_type = self.mcp_tool_definitions[server_name][
                    tool
                ].output_model

        if target_output_type and issubclass(target_output_type, BaseModel):
            if isinstance(result_data, dict):
                return target_output_type(**result_data)
            else:
                # If it's a primitive, try to wrap it if the model has one field
                fields = target_output_type.model_fields
                if len(fields) == 1:
                    field_name = list(fields.keys())[0]
                    return target_output_type(**{field_name: result_data})

        return result_data

    async def _call_python_tool(
        self,
        python_tool: str,
        arguments: Dict[str, Any],
        output_type: Optional[type] = None,
    ) -> Any:
        if python_tool in self.python_tools:
            func = self.python_tools[python_tool]
            definition = self.python_tool_definitions.get(python_tool)
        else:
            raise FunctionKernelException(
                f"Python tool '{python_tool}' not registered."
            )

        # Validate arguments using input model
        try:
            validated_input = definition.input_model(**arguments)
            call_args = validated_input.model_dump()

            # Ensure complex Pydantic types are passed as model instances if needed
            for param_name, p in inspect.signature(func).parameters.items():
                if (
                    param_name in call_args
                    and isinstance(p.annotation, type)
                    and issubclass(p.annotation, BaseModel)
                ):
                    call_args[param_name] = p.annotation(**call_args[param_name])
        except Exception as e:
            raise FunctionKernelException(
                f"Python tool '{python_tool}' argument validation failed: {e}"
            )

        sig = inspect.signature(func)
        try:
            bound_args = sig.bind(**call_args)
        except TypeError as e:
            raise FunctionKernelException(
                f"Python tool '{python_tool}' signature mismatch: {e}"
            )

        try:
            if inspect.iscoroutinefunction(func):
                result = await func(*bound_args.args, **bound_args.kwargs)
            else:
                result = func(*bound_args.args, **bound_args.kwargs)
        except Exception as e:
            logger.exception(f"Error executing python_tool '{python_tool}'")
            raise FunctionKernelException(
                f"Error executing python_tool '{python_tool}': {e}"
            )

        target_output_type = output_type or definition.output_model

        if target_output_type and issubclass(target_output_type, BaseModel):
            try:
                if isinstance(result, dict):
                    return target_output_type(**result)
                elif isinstance(result, BaseModel):
                    if isinstance(result, target_output_type):
                        return result
                    return target_output_type(**result.model_dump())
                else:
                    fields = target_output_type.model_fields
                    if len(fields) == 1:
                        field_name = list(fields.keys())[0]
                        return target_output_type(**{field_name: result})
                    else:
                        try:
                            return target_output_type(result)
                        except Exception:
                            return result
            except Exception as e:
                logger.warning(
                    f"Python tool '{python_tool}' returned incompatible result for {target_output_type}: {e}"
                )
                return result
        return result

    async def get_tool_descriptions(
        self, allowed_tools: Optional[List[str]] = None
    ) -> str:
        """Returns a string description of all registered tools as a JSON array for prompts."""
        tools_list = []

        def _get_input_schema(model: Type[BaseModel]) -> Dict[str, Any]:
            schema_dict = model.model_json_schema()
            # Remove pydantic-specific keys to keep it cleaner for LLM
            schema_dict.pop("title", None)
            schema_dict.pop("type", None)
            return schema_dict

        def _add_tool(name: str, description: str, input_model: Type[BaseModel]):
            if allowed_tools is not None and len(allowed_tools) > 0:
                # If we have an allowed list, only add if the tool is in it.
                # Note: name could be python://tool, rest://server.tool [METHOD], mcp://server.tool
                # We check for exact match or if it's a prefix for dynamic tools
                if name not in allowed_tools:
                    # Also check if it's a dynamic rest/mcp server and if the server is allowed
                    # e.g. rest://server.* or mcp://server.*
                    server_prefix = name.split(".")[0] + ".*"
                    if server_prefix not in allowed_tools:
                        return

            tools_list.append(
                {
                    "name": name,
                    "description": description,
                    "inputSchema": _get_input_schema(input_model),
                }
            )

        # Python tools
        for name, definition in self.python_tool_definitions.items():
            _add_tool(
                f"python://{name}",
                definition.description,
                definition.input_model,
            )

        # REST tools
        for server_name, tools in self.rest_tool_definitions.items():
            for tool_name, definition in tools.items():
                method = "GET"
                description_text = definition.description or ""
                try:
                    desc_data = json.loads(definition.description)
                    method = desc_data.get("method", "GET").upper()
                    description_text = desc_data.get("description", "")
                except Exception:
                    pass

                _add_tool(
                    f"rest://{server_name}.{tool_name} [{method}]",
                    description_text,
                    definition.input_model,
                )

        # REST servers - list those without specific tools registered
        for name in self.rest_servers.keys():
            if name not in self.rest_tool_definitions:
                tools_list.append(
                    {
                        "name": f"rest://{name}.<function_name>",
                        "description": f"Dynamic REST call to {name} server",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                )

        # MCP tools
        for server_name, tools in self.mcp_tool_definitions.items():
            for tool_name, definition in tools.items():
                _add_tool(
                    f"mcp://{server_name}.{tool_name}",
                    definition.description or "",
                    definition.input_model,
                )

        # Also list servers that might not have tools fetched yet
        for name in self.mcp_servers.keys():
            if name not in self.mcp_tool_definitions:
                tools_list.append(
                    {
                        "name": f"mcp://{name}.<tools_not_yet_loaded>",
                        "description": f"Dynamic MCP call to {name} server",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                )

        return json.dumps(tools_list, indent=2)
