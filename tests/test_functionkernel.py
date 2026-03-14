import pytest
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Any, Dict
from pydantic import BaseModel
from kavalai.functionkernel import FunctionKernel
from kavalai.agents.workflow_model import RestServer, McpServer, WorkflowException


class SimpleModel(BaseModel):
    name: str
    value: int


class CustomOutput(BaseModel):
    result: int
    meta: str


# Simple functions for testing
def sync_add(a: int, b: int) -> int:
    """Adds two integers."""
    return a + b


async def async_multiply(a: int, b: int) -> int:
    """Multiplies two integers asynchronously."""
    await asyncio.sleep(0.01)
    return a * b


def dict_output(name: str) -> Dict[str, Any]:
    return {"name": name, "value": 42}


def custom_model_output(val: int) -> CustomOutput:
    return CustomOutput(result=val * 2, meta="test")


@pytest.mark.asyncio
async def test_register_and_call_python_tool():
    kernel = FunctionKernel()
    kernel.register_python_tool("add", sync_add)
    kernel.register_python_tool("mul", async_multiply)

    # Sync call - returns model with 'result' field
    result = await kernel.call_python_tool("add", {"a": 1, "b": 2})
    assert result.result == 3

    # Async call
    result = await kernel.call_python_tool("mul", {"a": 3, "b": 4})
    assert result.result == 12

    # With output_type
    kernel.register_python_tool("dict_tool", dict_output)
    result = await kernel.call_python_tool(
        "dict_tool", {"name": "test"}, output_type=SimpleModel
    )
    assert isinstance(result, SimpleModel)
    assert result.name == "test"
    assert result.value == 42


@pytest.mark.asyncio
async def test_python_tool_custom_model():
    kernel = FunctionKernel()
    kernel.register_python_tool("custom", custom_model_output)

    definition = kernel.python_tool_definitions["custom"]
    assert definition.output_model == CustomOutput

    result = await kernel.call_python_tool("custom", {"val": 5})
    assert isinstance(result, CustomOutput)
    assert result.result == 10
    assert result.meta == "test"


@pytest.mark.asyncio
async def test_python_tool_errors():
    kernel = FunctionKernel()

    # Missing tool
    with pytest.raises(WorkflowException, match="Failed to load python_tool"):
        await kernel.call_python_tool("non.existent.tool", {})

    # Signature mismatch / validation error
    kernel.register_python_tool("add", sync_add)
    with pytest.raises(WorkflowException, match="argument validation failed"):
        await kernel.call_python_tool("add", {"a": 1})  # missing b


@pytest.mark.asyncio
async def test_python_tool_unregistered_load():
    kernel = FunctionKernel()
    # Should work via dynamic loading if module is accessible
    # Using full path to this test file's function
    tool_uri = "tests.test_functionkernel.sync_add"
    result = await kernel.call_python_tool(tool_uri, {"a": 10, "b": 20})
    assert result.result == 30


@pytest.mark.asyncio
async def test_call_rest_tool():
    kernel = FunctionKernel()
    server = RestServer(name="test_server", url="http://api.example.com")
    kernel.register_rest_server(server)

    with patch("httpx.AsyncClient.request") as mock_request:
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"name": "rest_test", "value": 100})
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        # Get request
        result = await kernel.call_rest_tool("test_server", "get_item", {"id": 1})
        assert result == {"name": "rest_test", "value": 100}
        mock_request.assert_called_with(
            "GET", "http://api.example.com/get_item", params={"id": 1}, timeout=60.0
        )

        # Post request with model output
        result = await kernel.call_rest_tool(
            "test_server",
            "create_item",
            {"name": "new"},
            method="post",
            output_type=SimpleModel,
        )
        assert isinstance(result, SimpleModel)
        assert result.name == "rest_test"
        mock_request.assert_called_with(
            "POST",
            "http://api.example.com/create_item",
            json={"name": "new"},
            timeout=60.0,
        )


@pytest.mark.asyncio
async def test_mcp_tool_stdio():
    kernel = FunctionKernel()
    server = McpServer(name="mcp_stdio", command="python", args=["mcp_server.py"])
    kernel.register_mcp_server(server)

    with patch("kavalai.functionkernel.stdio_client") as mock_stdio:
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_stdio.return_value.__aenter__.return_value = (mock_read, mock_write)

        with patch("kavalai.functionkernel.ClientSession") as mock_session_cls:
            mock_session = mock_session_cls.return_value
            mock_session.initialize = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()

            # Mock list_tools for initial registration
            mock_tool = MagicMock()
            mock_tool.name = "test_tool"
            mock_tool.description = "A test tool"
            mock_tool.inputSchema = {
                "type": "object",
                "properties": {"arg": {"type": "string"}},
            }
            mock_tools_result = MagicMock()
            mock_tools_result.tools = [mock_tool]
            mock_session.list_tools = AsyncMock(return_value=mock_tools_result)

            mock_response = MagicMock()
            mock_response.isError = False
            mock_content = MagicMock()
            mock_content.text = json.dumps({"name": "mcp_test", "value": 200})
            mock_response.content = [mock_content]

            # Mock response for the second call to match the default mcp_stdio_test_tool_output (which expects a 'result' field)
            mock_response_2 = MagicMock()
            mock_response_2.isError = False
            mock_content_2 = MagicMock()
            mock_content_2.text = json.dumps({"result": "any_value"})
            mock_response_2.content = [mock_content_2]

            mock_session.call_tool = AsyncMock(
                side_effect=[mock_response, mock_response_2]
            )

            result = await kernel.call_mcp_tool(
                "mcp_stdio", "test_tool", {"arg": "val"}, output_type=SimpleModel
            )

            assert isinstance(result, SimpleModel)
            assert result.name == "mcp_test"
            assert result.value == 200

            mock_session.initialize.assert_called_once()
            mock_session.call_tool.assert_called_with(
                "test_tool", arguments={"arg": "val"}
            )

            # Second call should reuse session
            await kernel.call_mcp_tool("mcp_stdio", "test_tool", {"arg": "val"})
            mock_session_cls.assert_called_once()

            # Cleanup
            await kernel.close()
            mock_stdio.return_value.__aexit__.assert_called()
            mock_session.__aexit__.assert_called()


@pytest.mark.asyncio
async def test_mcp_tool_sse():
    kernel = FunctionKernel()
    server = McpServer(name="mcp_sse", url="http://mcp.example.com/sse")
    kernel.register_mcp_server(server)

    with patch("kavalai.functionkernel.sse_client") as mock_sse:
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_sse.return_value.__aenter__.return_value = (mock_read, mock_write)

        with patch("kavalai.functionkernel.ClientSession") as mock_session_cls:
            mock_session = mock_session_cls.return_value
            mock_session.initialize = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()

            # Mock list_tools
            mock_tool = MagicMock()
            mock_tool.name = "hello"
            mock_tool.description = "Greet"
            mock_tool.inputSchema = {"type": "object", "properties": {}}
            mock_tools_result = MagicMock()
            mock_tools_result.tools = [mock_tool]
            mock_session.list_tools = AsyncMock(return_value=mock_tools_result)

            mock_response = MagicMock()
            mock_response.isError = False
            mock_content = MagicMock()
            mock_content.text = json.dumps({"result": "ok"})
            mock_response.content = [mock_content]
            mock_session.call_tool = AsyncMock(return_value=mock_response)

            result = await kernel.call_mcp_tool("mcp_sse", "hello", {})
            assert result.result == "ok"
            mock_sse.assert_called_with("http://mcp.example.com/sse")


@pytest.mark.asyncio
async def test_tool_descriptions():
    kernel = FunctionKernel()
    kernel.register_python_tool("math.add", sync_add)
    kernel.register_rest_server(RestServer(name="my_rest", url="http://api.com"))
    kernel.register_mcp_server(McpServer(name="my_mcp", command="ls"))

    with patch("kavalai.functionkernel.stdio_client") as mock_stdio:
        mock_stdio.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock())
        with patch("kavalai.functionkernel.ClientSession") as mock_session_cls:
            mock_session = mock_session_cls.return_value
            mock_session.initialize = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)

            mock_tool = MagicMock()
            mock_tool.name = "list_files"
            mock_tool.description = "Lists files in directory"
            mock_tool.inputSchema = {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            }

            mock_tools_result = MagicMock()
            mock_tools_result.tools = [mock_tool]
            mock_session.list_tools = AsyncMock(return_value=mock_tools_result)

            # Pre-initialize MCP session to load tools
            await kernel._get_mcp_session("my_mcp")

            desc = await kernel.get_tool_descriptions()

            assert "python://math.add" in desc
            assert "rest://my_rest.<function_name>" in desc
            assert "mcp://my_mcp.list_files" in desc

            # Verify Pydantic schemas are present
            assert "Input" in desc
            assert "Output" in desc
            assert "integer" in desc


@pytest.mark.asyncio
async def test_call_tool_unified():
    kernel = FunctionKernel()

    # Test Python tool via unified call
    kernel.register_python_tool("math.add", sync_add)
    result = await kernel.call_tool("python://math.add", {"a": 10, "b": 20})
    assert result.result == 30

    # Test REST tool via unified call
    server = RestServer(name="test_api", url="http://api.example.com")
    kernel.register_rest_server(server)

    with patch("httpx.AsyncClient.request") as mock_request:
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"status": "ok"})
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        result = await kernel.call_tool("rest://test_api.status", {})
        assert result == {"status": "ok"}
        mock_request.assert_called_with(
            "GET", "http://api.example.com/status", params={}, timeout=60.0
        )

    # Test MCP tool via unified call
    mcp_server = McpServer(name="test_mcp", command="true")
    kernel.register_mcp_server(mcp_server)

    with patch("kavalai.functionkernel.stdio_client") as mock_stdio:
        mock_stdio.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock())
        with patch("kavalai.functionkernel.ClientSession") as mock_session_cls:
            mock_session = mock_session_cls.return_value
            mock_session.initialize = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)

            # Mock list_tools
            mock_tool = MagicMock()
            mock_tool.name = "test_tool"
            mock_tool.description = "A test tool"
            mock_tool.inputSchema = {
                "type": "object",
                "properties": {"arg": {"type": "integer"}},
            }
            mock_tools_result = MagicMock()
            mock_tools_result.tools = [mock_tool]
            mock_session.list_tools = AsyncMock(return_value=mock_tools_result)

            mock_response = MagicMock()
            mock_response.isError = False
            mock_content = MagicMock()
            mock_content.text = json.dumps({"result": {"mcp": "works"}})
            mock_response.content = [mock_content]
            mock_session.call_tool = AsyncMock(return_value=mock_response)

            result = await kernel.call_tool("mcp://test_mcp.test_tool", {"arg": 1})
            assert result.result == {"mcp": "works"}
            mock_session.call_tool.assert_called_with("test_tool", arguments={"arg": 1})


@pytest.mark.asyncio
async def test_call_tool_errors():
    kernel = FunctionKernel()

    with pytest.raises(WorkflowException, match="Invalid tool URI format"):
        await kernel.call_tool("invalid_format", {})

    with pytest.raises(WorkflowException, match="Invalid tool path format"):
        await kernel.call_tool("python://nodot", {})

    with pytest.raises(WorkflowException, match="Unsupported protocol"):
        await kernel.call_tool("ftp://some.file", {})


@pytest.mark.asyncio
async def test_registration_conflicts():
    kernel = FunctionKernel()

    # Test REST server conflict
    kernel.register_rest_server(RestServer(name="test", url="http://api.com"))
    with pytest.raises(
        WorkflowException, match="REST server 'test' is already registered"
    ):
        kernel.register_rest_server(RestServer(name="test", url="http://other.com"))

    # Test MCP server conflict
    kernel.register_mcp_server(McpServer(name="mcp", command="ls"))
    with pytest.raises(
        WorkflowException, match="MCP server 'mcp' is already registered"
    ):
        kernel.register_mcp_server(McpServer(name="mcp", command="dir"))

    # Test Python tool conflict
    kernel.register_python_tool("add", sync_add)
    with pytest.raises(
        WorkflowException, match="Python tool 'add' is already registered"
    ):
        kernel.register_python_tool("add", async_multiply)
