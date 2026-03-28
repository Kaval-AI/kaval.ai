from kavalai.agents.workflow_model import to_plain
from kavalai.functionkernel import FunctionKernel, pythontool
import pytest

def test_to_plain_filters_metadata():
    data = {
        "name": "task",
        "__line__": 10,
        "__file_path__": "/path/to/file.yaml",
        "nested": {
            "value": 1,
            "__line__": 11
        },
        "list": [
            {"item": 1, "__line__": 12}
        ]
    }
    
    expected = {
        "name": "task",
        "nested": {
            "value": 1
        },
        "list": [
            {"item": 1}
        ]
    }
    
    assert to_plain(data) == expected

@pytest.mark.asyncio
async def test_call_tool_strips_metadata():
    kernel = FunctionKernel()
    
    @pythontool
    def my_tool(a: int, b: int):
        return a + b
        
    kernel.register_python_tool("my_tool", my_tool)
    
    # Pass metadata that should be stripped
    result = await kernel.call_tool("python://my_tool", {"a": 1, "b": 2, "__line__": 99, "__file_path__": "test.yaml"})
    
    assert result.result == 3
