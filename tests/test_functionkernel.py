import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from kavalai.functionkernel import FunctionKernel
from kavalai.agents.workflow_model import RestServer, McpServer, WorkflowException
from pydantic import BaseModel
import json


class SimpleModel(BaseModel):
    name: str
    value: int


def sync_add(a: int, b: int):
    """Adds two integers."""
    return a + b


async def async_multiply(a: int, b: int):
    """Multiplies two integers asynchronously."""
    return a * b


def dict_output(name: str):
    return {"name": name, "value": 42}


@pytest.mark.asyncio
async def test_register_and_call_python_tool():
    kernel = FunctionKernel()
    kernel.register_python_tool("add", sync_add)
    kernel.register_python_tool("mul", async_multiply)

    # Sync call
    result = await kernel.call_python_tool("add", {"a": 1, "b": 2})
    assert result == 3

    # Async call
    result = await kernel.call_python_tool("mul", {"a": 3, "b": 4})
    assert result == 12

    # With output_type
    kernel.register_python_tool("dict_tool", dict_output)
    result = await kernel.call_python_tool(
        "dict_tool", {"name": "test"}, output_type=SimpleModel
    )
    assert isinstance(result, SimpleModel)
    assert result.name == "test"
    assert result.value == 42


@pytest.mark.asyncio
async def test_python_tool_errors():
    kernel = FunctionKernel()

    # Missing tool
    with pytest.raises(WorkflowException, match="Failed to load python_tool"):
        await kernel.call_python_tool("non.existent.tool", {})

    # Signature mismatch
    kernel.register_python_tool("add", sync_add)
    with pytest.raises(WorkflowException, match="signature mismatch"):
        await kernel.call_python_tool("add", {"a": 1})  # missing b


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

            mock_response = MagicMock()
            mock_response.isError = False
            mock_content = MagicMock()
            mock_content.text = json.dumps({"name": "mcp_test", "value": 200})
            mock_response.content = [mock_content]
            mock_session.call_tool = AsyncMock(return_value=mock_response)

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

            mock_response = MagicMock()
            mock_response.isError = False
            mock_content = MagicMock()
            mock_content.text = json.dumps({"result": "ok"})
            mock_response.content = [mock_content]
            mock_session.call_tool = AsyncMock(return_value=mock_response)

            result = await kernel.call_mcp_tool("mcp_sse", "hello", {})
            assert result == {"result": "ok"}
            mock_sse.assert_called_with("http://mcp.example.com/sse")


def test_tool_descriptions():
    kernel = FunctionKernel()
    kernel.register_python_tool("add", sync_add)
    kernel.register_rest_server(RestServer(name="my_rest", url="http://api.com"))
    kernel.register_mcp_server(McpServer(name="my_mcp", command="ls"))

    desc = kernel.get_tool_descriptions()
    assert "Python Tool: add(a: int, b: int)" in desc
    assert "Description: Adds two integers." in desc
    assert "REST Server: my_rest" in desc
    assert "MCP Server: my_mcp" in desc
