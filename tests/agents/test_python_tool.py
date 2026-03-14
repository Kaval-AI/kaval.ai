import pytest
import asyncio
from kavalai.agents.workflow import Workflow
from kavalai.agents.workflow_model import WorkflowException
from kavalai.functionkernel import pythontool


# Simple functions for testing
@pythontool
def sync_add(a: int, b: int) -> int:
    return a + b


@pythontool
async def async_multiply(x: float, y: float) -> float:
    await asyncio.sleep(0.01)
    return x * y


@pythontool
def dict_output(name: str) -> dict:
    return {"message": f"Hello {name}"}


@pytest.mark.asyncio
async def test_python_tool_sync():
    yaml_content = """
name: test_python_tool
data_types:
  input:
    type: object
    properties:
      a: {type: integer}
      b: {type: integer}
  output:
    type: object
    properties:
      result: {type: integer}
tasks:
  - name: add_task
    inputs:
      a: {type: context, value: input.a}
      b: {type: context, value: input.b}
    type: python
    python_tool: sync_add
    output: output
python_functions:
  - name: sync_add
    path: tests.agents.test_python_tool.sync_add
"""
    workflow = Workflow.from_yaml(yaml_content)
    result = await workflow.run({"a": 10, "b": 20})
    assert result.data.result == 30


@pytest.mark.asyncio
async def test_python_tool_async():
    yaml_content = """
name: test_python_tool_async
data_types:
  input:
    type: object
    properties:
      x: {type: number}
      y: {type: number}
  output:
    type: object
    properties:
      result: {type: number}
tasks:
  - name: multiply_task
    inputs:
      x: {type: context, value: input.x}
      y: {type: context, value: input.y}
    type: python
    python_tool: async_multiply
    output: output
python_functions:
  - name: async_multiply
    path: tests.agents.test_python_tool.async_multiply
"""
    workflow = Workflow.from_yaml(yaml_content)
    result = await workflow.run({"x": 2.5, "y": 4.0})
    assert result.data.result == 10.0


@pytest.mark.asyncio
async def test_python_tool_dict_output():
    yaml_content = """
name: test_python_tool_dict
data_types:
  input:
    type: object
    properties:
      user_name: {type: string}
  output:
    type: object
    properties:
      message: {type: string}
tasks:
  - name: greet_task
    inputs:
      name: {type: context, value: input.user_name}
    type: python
    python_tool: dict_output
    output: output
python_functions:
  - name: dict_output
    path: tests.agents.test_python_tool.dict_output
"""
    workflow = Workflow.from_yaml(yaml_content)
    result = await workflow.run({"user_name": "World"})
    assert result.data.message == "Hello World"


@pytest.mark.asyncio
async def test_python_tool_error_loading():
    yaml_content = """
name: test_error
data_types:
  input: {type: object, properties: {}}
  output:
    type: object
    properties:
      res: {type: string}
tasks:
  - name: fail_task
    type: python
    python_tool: non_existent
    output: output
python_functions:
  - name: non_existent
    path: non_existent_module.func
"""
    with pytest.raises(
        (WorkflowException, ImportError, AttributeError, ModuleNotFoundError)
    ):
        Workflow.from_yaml(yaml_content)


@pytest.mark.asyncio
async def test_python_tool_signature_mismatch():
    yaml_content = """
name: test_sig_error
data_types:
  input:
    type: object
    properties:
      a: {type: integer}
  output:
    type: object
    properties:
      res: {type: integer}
tasks:
  - name: fail_task
    inputs:
      a: {type: context, value: input.a}
    # sync_add needs a and b
    type: python
    python_tool: sync_add
    output: output
python_functions:
  - name: sync_add
    path: tests.agents.test_python_tool.sync_add
"""
    workflow = Workflow.from_yaml(yaml_content)
    with pytest.raises(WorkflowException) as excinfo:
        await workflow.run({"a": 1})
    # FunctionKernel validates arguments via Pydantic, so error message is different
    assert "argument validation failed" in str(
        excinfo.value
    ) or "signature mismatch" in str(excinfo.value)


@pytest.mark.asyncio
async def test_python_tool_multi_field_output():
    @pythontool
    def multi_field(a: int) -> dict:
        return {"res1": a, "res2": a * 2}

    global multi_field_test
    multi_field_test = multi_field

    yaml_content = """
name: test_multi_field
data_types:
  input: {type: object, properties: {a: {type: integer}}}
  output:
    type: object
    properties:
      res1: {type: integer}
      res2: {type: integer}
tasks:
  - name: task1
    inputs: {a: {type: context, value: input.a}}
    type: python
    python_tool: multi_field_test
    output: output
python_functions:
  - name: multi_field_test
    path: tests.agents.test_python_tool.multi_field_test
"""
    workflow = Workflow.from_yaml(yaml_content)
    result = await workflow.run({"a": 5})
    assert result.data.res1 == 5
    assert result.data.res2 == 10


@pytest.mark.asyncio
async def test_python_tool_multi_field_mismatch():
    @pythontool
    def multi_field(a: int) -> int:
        return a

    global multi_field_mismatch_test
    multi_field_mismatch_test = multi_field

    yaml_content = """
name: test_multi_field_mismatch
data_types:
  input: {type: object, properties: {a: {type: integer}}}
  output:
    type: object
    properties:
      res1: {type: integer}
      res2: {type: integer}
tasks:
  - name: task1
    inputs: {a: {type: context, value: input.a}}
    type: python
    python_tool: multi_field_mismatch_test
    output: output
python_functions:
  - name: multi_field_mismatch_test
    path: tests.agents.test_python_tool.multi_field_mismatch_test
"""
    workflow = Workflow.from_yaml(yaml_content)
    # FunctionKernel returns raw result when conversion fails, which then fails validation
    # in WorkflowRunResult, so we catch any validation error
    with pytest.raises((WorkflowException, Exception)) as excinfo:
        await workflow.run({"a": 5})
    # The error can be about incompatible result or validation error
    error_str = str(excinfo.value)
    assert (
        "returned incompatible result" in error_str
        or "validation error" in error_str.lower()
        or "ValidationError" in str(type(excinfo.value))
    )


@pytest.mark.asyncio
async def test_python_tool_execution_error():
    @pythontool
    def error_func():
        raise ValueError("Something went wrong")

    global error_func_test
    error_func_test = error_func

    yaml_content = """
name: test_exec_error
data_types:
  input: {type: object, properties: {}}
  output:
    type: object
    properties:
      res: {type: integer}
tasks:
  - name: fail_task
    type: python
    python_tool: error_func_test
    output: output
python_functions:
  - name: error_func_test
    path: tests.agents.test_python_tool.error_func_test
"""
    workflow = Workflow.from_yaml(yaml_content)
    with pytest.raises(WorkflowException) as excinfo:
        await workflow.run({})
    assert "Error executing python_tool" in str(excinfo.value)
    assert "Something went wrong" in str(excinfo.value)
