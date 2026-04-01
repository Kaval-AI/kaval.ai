"""Tests for workflow.py REST server environment variable handling."""

from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
import json
from uuid import uuid4

import pytest
from pydantic import BaseModel, ValidationError

from kavalai.agents.workflow import Workflow
from kavalai.llm_clients.llm_client import LLMClient
from kavalai.agents.run_context import RunContext
from kavalai.agents.agent_service import AgentService
from kavalai.llm_clients.common import StreamContent
from kavalai.agents.workflow_model import (
    WorkflowModel,
    WorkflowException,
    RestServer,
    RestTask,
    ArgumentInfo,
)
from kavalai.agents.utils import to_plain


def create_workflow_model_with_rest_server(
    username_env: str = None, password_env: str = None, method: str = "get"
) -> WorkflowModel:
    """Helper to create a WorkflowModel with a REST server."""
    rest_servers = [
        RestServer(
            name="test_server",
            url="http://localhost:8000",
            username_env=username_env,
            password_env=password_env,
        )
    ]
    return WorkflowModel(
        name="test_workflow",
        description="Test workflow",
        data_types={
            "input": {"type": "object", "properties": {"query": {"type": "string"}}},
            "output": {"type": "object", "properties": {"result": {"type": "string"}}},
        },
        rest_servers=rest_servers,
        tasks=[
            RestTask(
                name="test_task",
                inputs={"query": ArgumentInfo(type="context", name="input")},
                output="output",
                tool="test_tool",
                rest_server="test_server",
                method=method,
            )
        ],
    )


class TestRunToolMethod:
    """Tests for run_tool method handling."""

    @pytest.mark.asyncio
    async def test_run_rest_tool_uses_default_get_method(self):
        """run_rest_tool should use GET method by default."""
        model = create_workflow_model_with_rest_server()
        workflow = Workflow(model)

        captured_tool_uri = None
        captured_method = None
        captured_arguments = None

        async def mock_call_tool(tool_uri, arguments, output_type=None, **kwargs):
            nonlocal captured_tool_uri, captured_method, captured_arguments
            captured_tool_uri = tool_uri
            captured_method = kwargs.get("method", "get")
            captured_arguments = arguments
            # Return a mock output matching the expected type
            return workflow.get_data_type("output")(result="success")

        workflow.kernel.call_tool = mock_call_tool

        task = workflow.workflow_model.tasks[0]
        run_context = RunContext()
        run_context.data["input"] = MagicMock()
        await workflow.run_rest_tool(task, run_context)

        assert captured_tool_uri == "rest://test_server.test_tool"
        assert captured_method == "get"

    @pytest.mark.asyncio
    async def test_run_rest_tool_uses_specified_post_method(self):
        """run_rest_tool should use POST method when specified."""
        model = create_workflow_model_with_rest_server(method="post")
        workflow = Workflow(model)

        captured_tool_uri = None
        captured_method = None
        captured_arguments = None

        async def mock_call_tool(tool_uri, arguments, output_type=None, **kwargs):
            nonlocal captured_tool_uri, captured_method, captured_arguments
            captured_tool_uri = tool_uri
            captured_method = kwargs.get("method", "get")
            captured_arguments = arguments
            # Return a mock output matching the expected type
            return workflow.get_data_type("output")(result="success")

        workflow.kernel.call_tool = mock_call_tool

        task = workflow.workflow_model.tasks[0]
        run_context = RunContext()
        run_context.data["input"] = MagicMock()
        await workflow.run_rest_tool(task, run_context)

        assert captured_tool_uri == "rest://test_server.test_tool"
        assert captured_method == "post"


class TestRestToolSerialization:
    @pytest.mark.asyncio
    async def test_run_rest_tool_serializes_pydantic_models(self):
        """run_rest_tool should serialize Pydantic models to dicts in JSON body."""

        class TestModel(BaseModel):
            name: str

        # Define a workflow model with a POST task
        model = WorkflowModel(
            name="Test Workflow",
            data_types={
                "test_model": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
                "output_model": {
                    "type": "object",
                    "properties": {"result": {"type": "string"}},
                },
            },
            rest_servers=[RestServer(name="test_server", url="http://localhost:1234")],
            tasks=[
                RestTask(
                    name="Task 1",
                    tool="search",
                    rest_server="test_server",
                    method="post",
                    inputs={"input": ArgumentInfo(type="context", value="input")},
                    output="output_model",
                )
            ],
        )
        workflow = Workflow(model)

        captured_arguments = None

        async def mock_call_tool(tool_uri, arguments, output_type=None, **kwargs):
            nonlocal captured_arguments
            captured_arguments = arguments
            # Return a mock output matching the expected type
            return workflow.get_data_type("output_model")(result="success")

        workflow.kernel.call_tool = mock_call_tool

        run_context = RunContext()
        run_context.data["input"] = TestModel(name="test_value")
        await workflow.run_rest_tool(workflow.workflow_model.tasks[0], run_context)

        # Check that the Pydantic model was converted to a dict
        # After changing prepare_tool_inputs to always return a dict,
        # single inputs are now wrapped in a dict with the input name as key.
        assert captured_arguments == {"input": {"name": "test_value"}}
        assert isinstance(captured_arguments, dict)


class TestRestServerModelValidation:
    """Tests for RestServer Pydantic model validation."""

    def test_rest_server_must_have_url_or_url_env(self):
        with pytest.raises(ValidationError) as exc_info:
            RestServer(name="test_server")
        assert "Either 'url' or 'url_env' must be specified" in str(exc_info.value)

    def test_rest_server_cannot_have_both_url_and_url_env(self):
        with pytest.raises(ValidationError) as exc_info:
            RestServer(
                name="test_server", url="http://localhost:8000", url_env="TEST_URL_ENV"
            )
        assert "Only one of 'url' or 'url_env' can be specified" in str(exc_info.value)


class TestRestServerEnvVarValidation:
    """Tests for REST server environment variable validation during workflow loading."""

    def test_workflow_loads_without_auth_env_vars(self):
        """Workflow should load successfully when no auth env vars are specified."""
        model = create_workflow_model_with_rest_server()
        workflow = Workflow(model)
        assert workflow is not None
        assert "test_server" in workflow.kernel.rest_servers

    def test_workflow_loads_with_url_env(self, monkeypatch):
        monkeypatch.setenv("MY_REST_URL", "http://my-api.com")
        # create_workflow_model_with_rest_server uses url by default
        # We need to create a model where only url_env is specified
        rest_servers = [RestServer(name="test_server", url_env="MY_REST_URL")]
        model = create_workflow_model_with_rest_server()
        model.rest_servers = rest_servers

        workflow = Workflow(model)
        # url should stay None
        assert workflow.kernel.rest_servers["test_server"].url is None
        # url_env should stay as specified
        assert workflow.kernel.rest_servers["test_server"].url_env == "MY_REST_URL"

    def test_workflow_raises_on_invalid_url_env_value(self, monkeypatch):
        """Workflow should raise exception when url_env value is invalid."""
        monkeypatch.setenv("MY_REST_URL", "not-a-url")
        rest_servers = [RestServer(name="test_server", url_env="MY_REST_URL")]
        model = create_workflow_model_with_rest_server()
        model.rest_servers = rest_servers

        with pytest.raises(WorkflowException) as exc_info:
            Workflow(model)
        assert "invalid URL" in str(exc_info.value)
        assert "from MY_REST_URL" in str(exc_info.value)

    def test_workflow_raises_on_missing_url_env(self, monkeypatch):
        """Workflow should raise exception when url_env is not defined."""
        monkeypatch.delenv("MY_REST_URL", raising=False)
        rest_servers = [RestServer(name="test_server", url_env="MY_REST_URL")]
        model = create_workflow_model_with_rest_server()
        model.rest_servers = rest_servers

        with pytest.raises(WorkflowException) as exc_info:
            Workflow(model)
        assert "MY_REST_URL" in str(exc_info.value)
        assert "URL is not defined" in str(exc_info.value)

    def test_workflow_raises_on_invalid_url(self):
        """Workflow should raise exception when static URL is invalid."""
        model = create_workflow_model_with_rest_server()
        model.rest_servers[0].url = "not-a-url"

        with pytest.raises(WorkflowException) as exc_info:
            Workflow(model)
        assert "invalid URL" in str(exc_info.value)

    def test_workflow_loads_with_valid_auth_env_vars(self, monkeypatch):
        """Workflow should load when both env vars are defined."""
        monkeypatch.setenv("TEST_USERNAME", "user123")
        monkeypatch.setenv("TEST_PASSWORD", "pass456")

        model = create_workflow_model_with_rest_server(
            username_env="TEST_USERNAME", password_env="TEST_PASSWORD"
        )
        workflow = Workflow(model)
        assert workflow is not None

    def test_workflow_raises_when_username_env_missing(self, monkeypatch):
        """Workflow should raise exception when username env var is not defined."""
        monkeypatch.setenv("TEST_PASSWORD", "pass456")
        # Ensure TEST_USERNAME is not set
        monkeypatch.delenv("TEST_USERNAME", raising=False)

        model = create_workflow_model_with_rest_server(
            username_env="TEST_USERNAME", password_env="TEST_PASSWORD"
        )
        with pytest.raises(WorkflowException) as exc_info:
            Workflow(model)
        assert "TEST_USERNAME" in str(exc_info.value)
        assert "username is not defined" in str(exc_info.value)

    def test_workflow_raises_when_password_env_missing(self, monkeypatch):
        """Workflow should raise exception when password env var is not defined."""
        monkeypatch.setenv("TEST_USERNAME", "user123")
        # Ensure TEST_PASSWORD is not set
        monkeypatch.delenv("TEST_PASSWORD", raising=False)

        model = create_workflow_model_with_rest_server(
            username_env="TEST_USERNAME", password_env="TEST_PASSWORD"
        )
        with pytest.raises(WorkflowException) as exc_info:
            Workflow(model)
        assert "TEST_PASSWORD" in str(exc_info.value)
        assert "password is not defined" in str(exc_info.value)

    def test_workflow_raises_when_only_username_env_defined(self, monkeypatch):
        """Workflow should raise exception when only username_env is specified."""
        monkeypatch.setenv("TEST_USERNAME", "user123")

        model = create_workflow_model_with_rest_server(
            username_env="TEST_USERNAME", password_env=None
        )
        with pytest.raises(WorkflowException) as exc_info:
            Workflow(model)
        assert "must have both username_env and password_env" in str(exc_info.value)

    def test_workflow_raises_when_only_password_env_defined(self, monkeypatch):
        """Workflow should raise exception when only password_env is specified."""
        monkeypatch.setenv("TEST_PASSWORD", "pass456")

        model = create_workflow_model_with_rest_server(
            username_env=None, password_env="TEST_PASSWORD"
        )
        with pytest.raises(WorkflowException) as exc_info:
            Workflow(model)
        assert "must have both username_env and password_env" in str(exc_info.value)


class TestRunRestToolAuth:
    """Tests for run_rest_tool method authentication handling."""

    @pytest.mark.asyncio
    async def test_run_rest_tool_uses_basic_auth_when_env_vars_defined(
        self, monkeypatch
    ):
        """run_rest_tool should use FunctionKernel with servers that have auth configured."""
        monkeypatch.setenv("TEST_USERNAME", "user123")
        monkeypatch.setenv("TEST_PASSWORD", "pass456")

        model = create_workflow_model_with_rest_server(
            username_env="TEST_USERNAME", password_env="TEST_PASSWORD"
        )
        workflow = Workflow(model)

        # Verify that the REST server is registered in FunctionKernel with auth env vars
        assert "test_server" in workflow.kernel.rest_servers
        assert (
            workflow.kernel.rest_servers["test_server"].username_env == "TEST_USERNAME"
        )
        assert (
            workflow.kernel.rest_servers["test_server"].password_env
            == "TEST_PASSWORD"  # gitleaks:allow
        )

        # Mock the kernel's call_tool to verify it's called correctly
        async def mock_call_tool(tool_uri, arguments, output_type=None, **kwargs):
            return workflow.get_data_type("output")(result="success")

        workflow.kernel.call_tool = mock_call_tool

        task = workflow.workflow_model.tasks[0]
        run_context = RunContext()
        run_context.data["input"] = MagicMock()
        await workflow.run_rest_tool(task, run_context)

        # Test passes if no exception is raised

    @pytest.mark.asyncio
    async def test_run_rest_tool_no_auth_when_env_vars_not_defined(self, monkeypatch):
        """run_rest_tool should use FunctionKernel with servers that have no auth configured."""
        model = create_workflow_model_with_rest_server()
        workflow = Workflow(model)

        # Verify that the REST server is registered in FunctionKernel without auth env vars
        assert "test_server" in workflow.kernel.rest_servers
        assert workflow.kernel.rest_servers["test_server"].username_env is None
        assert workflow.kernel.rest_servers["test_server"].password_env is None

        # Mock the kernel's call_tool to verify it's called correctly
        async def mock_call_tool(tool_uri, arguments, output_type=None, **kwargs):
            return workflow.get_data_type("output")(result="success")

        workflow.kernel.call_tool = mock_call_tool

        task = workflow.workflow_model.tasks[0]
        run_context = RunContext()
        run_context.data["input"] = MagicMock()
        await workflow.run_rest_tool(task, run_context)

        # Test passes if no exception is raised


class TestWorkflowFeatures:
    def test_resolve_context_value_nested(self):
        from kavalai.agents.run_context import RunContext
        from pydantic import BaseModel

        class NestedModel(BaseModel):
            field: str

        class RootModel(BaseModel):
            nested: NestedModel
            value: int

        data = {
            "input": RootModel(nested=NestedModel(field="hello"), value=42),
            "simple": {"key": "val"},
        }
        ctx = RunContext(data=data)

        assert ctx.resolve_context_value("input.value") == 42
        assert ctx.resolve_context_value("input.nested.field") == "hello"
        assert ctx.resolve_context_value("simple.key") == "val"
        assert ctx.resolve_context_value("nonexistent") is None
        assert ctx.resolve_context_value("input.nonexistent") is None

    @pytest.mark.asyncio
    async def test_workflow_run_combine(self):
        from kavalai.agents.workflow import Workflow

        yaml_content = """
name: Test Workflow
data_types:
  input:
    type: object
    properties:
      msg: { type: string }
  output:
    type: object
    properties:
      final_msg: { type: string }
tasks:
  - name: Combine
    type: combine
    inputs:
      final_msg: { value: input.msg, type: context }
    output: output
"""
        w = Workflow.from_yaml(yaml_content)
        w.agent_service = None

        result = await w.run({"msg": "hello"})
        assert getattr(result.data, "final_msg") == "hello"

    @pytest.mark.asyncio
    async def test_workflow_run_combine_named_type(self):
        yaml_content = """
name: Test Workflow
data_types:
  input:
    type: object
    properties:
      msg: { type: string }
  intermediate:
    type: object
    properties:
      text: { type: string }
  output:
    type: object
    properties:
      final_msg: { type: string }
tasks:
  - name: Combine Intermediate
    type: combine
    inputs:
      text: { value: input.msg, type: context }
    output: intermediate
  - name: Final Output
    type: combine
    inputs:
      final_msg: { value: intermediate.text, type: context }
    output: output
"""
        w = Workflow.from_yaml(yaml_content)
        w.agent_service = None

        result = await w.run({"msg": "hello world"})
        assert getattr(result.data, "final_msg") == "hello world"


class TestWorkflowConditions:
    @pytest.mark.asyncio
    async def test_workflow_conditional_tasks(self):
        workflow_yaml = """
name: ConditionalTest
data_types:
  input:
    type: object
    properties:
      run_task:
        type: boolean
      value:
        type: integer
  output:
    type: object
    properties:
      result:
        type: string
tasks:
  - name: Conditional Task
    type: combine
    when:
      eq: [ { type: context, value: input.run_task }, true ]
    inputs:
      result:
        type: literal
        value: "Task executed"
    output: output
  - name: GT Task
    type: combine
    when:
      gt: [ { type: context, value: input.value }, 10 ]
    inputs:
      result:
        type: literal
        value: "Value is large"
    output: output
  - name: All Task
    type: combine
    when:
      all:
        - eq: [ { type: context, value: input.run_task }, true ]
        - gt: [ { type: context, value: input.value }, 5 ]
    inputs:
      result:
        type: literal
        value: "All conditions met"
    output: output
"""
        workflow = Workflow.from_yaml(workflow_yaml)

        # Case 1: run_task=False, value=5 -> No tasks should run, output should be empty/None
        workflow = Workflow.from_yaml(workflow_yaml)
        result1 = await workflow.run({"run_task": False, "value": 5})
        assert result1.data is None

        # Case 2: run_task=True, value=5 -> Only "Conditional Task" should run
        workflow = Workflow.from_yaml(workflow_yaml)
        result2 = await workflow.run({"run_task": True, "value": 5})
        assert result2.data.result == "Task executed"

        # Case 3: run_task=False, value=15 -> Only "GT Task" should run
        workflow = Workflow.from_yaml(workflow_yaml)
        result3 = await workflow.run({"run_task": False, "value": 15})
        assert result3.data.result == "Value is large"

        # Case 4: run_task=True, value=15 -> All tasks run, last one wins for "result"
        workflow = Workflow.from_yaml(workflow_yaml)
        result4 = await workflow.run({"run_task": True, "value": 15})
        assert result4.data.result == "All conditions met"

    @pytest.mark.asyncio
    async def test_workflow_conditional_any_not(self):
        workflow_yaml = """
name: AnyNotTest
data_types:
  input:
    type: object
    properties:
      val:
        type: integer
  output:
    type: object
    properties:
      result:
        type: string
tasks:
  - name: Any Task
    type: combine
    when:
      any:
        - eq: [ { type: context, value: input.val }, 1 ]
        - eq: [ { type: context, value: input.val }, 2 ]
    inputs:
      result:
        type: literal
        value: "One or Two"
    output: output
  - name: Not Task
    type: combine
    when:
      not:
        eq: [ { type: context, value: input.val }, 1 ]
    inputs:
      result:
        type: literal
        value: "Not One"
    output: output
"""
        workflow = Workflow.from_yaml(workflow_yaml)

        # val=1 -> "Any Task" runs, "Not Task" does not. Result: "One or Two"
        workflow = Workflow.from_yaml(workflow_yaml)
        result1 = await workflow.run({"val": 1})
        assert result1.data.result == "One or Two"

        # val=2 -> "Any Task" runs, "Not Task" runs. Result: "Not One"
        workflow = Workflow.from_yaml(workflow_yaml)
        result2 = await workflow.run({"val": 2})
        assert result2.data.result == "Not One"

        # val=3 -> "Any Task" does not run, "Not Task" runs. Result: "Not One"
        workflow = Workflow.from_yaml(workflow_yaml)
        result3 = await workflow.run({"val": 3})
        assert result3.data.result == "Not One"

    @pytest.mark.asyncio
    async def test_workflow_conditional_nested_paths(self):
        workflow_yaml = """
name: NestedPathTest
data_types:
  input:
    type: object
    properties:
      criteria:
        type: object
        properties:
          keywords:
            type: array
            items:
              type: string
  output:
    type: object
    properties:
      result:
        type: string
tasks:
  - name: Search
    type: combine
    when:
      gt: [ { type: context, value: "input.criteria.keywords.length" }, 0 ]
    inputs:
      result:
        type: literal
        value: "Found keywords"
    output: output
"""
        workflow = Workflow.from_yaml(workflow_yaml)

        # Empty keywords
        result1 = await workflow.run({"criteria": {"keywords": []}})
        assert result1.data is None

        # Some keywords
        result2 = await workflow.run({"criteria": {"keywords": ["hotel", "paris"]}})
        assert result2.data.result == "Found keywords"

    @pytest.mark.asyncio
    async def test_workflow_conditional_contains(self):
        workflow_yaml = """
name: ContainsTest
data_types:
  input:
    type: object
    properties:
      tags:
        type: array
        items:
          type: string
  output:
    type: object
    properties:
      result:
        type: string
tasks:
  - name: Tag Check
    type: combine
    when:
      contains: [ { type: context, value: "input.tags" }, "vip" ]
    inputs:
      result:
        type: literal
        value: "VIP Customer"
    output: output
"""
        workflow = Workflow.from_yaml(workflow_yaml)

        # No vip tag
        result1 = await workflow.run({"tags": ["new", "newsletter"]})
        assert result1.data is None

        # With vip tag
        result2 = await workflow.run({"tags": ["vip", "newsletter"]})
        assert result2.data.result == "VIP Customer"

    @pytest.mark.asyncio
    async def test_workflow_conditional_invalid_operator_length(self):
        workflow_yaml = """
name: InvalidLengthTest
data_types:
  input:
    type: object
    properties:
      val:
        type: integer
  output:
    type: object
    properties:
      result:
        type: string
tasks:
  - name: Invalid Task
    type: combine
    when:
      eq: [ { type: context, value: input.val }, 1 , 2]
    inputs:
      result:
        type: literal
        value: "Should fail"
    output: output
"""
        # The error now happens during Workflow.from_yaml (initialization) due to Pydantic validator
        with pytest.raises(
            WorkflowException, match="Operator 'eq' requires a list of 2 operands."
        ):
            Workflow.from_yaml(workflow_yaml)

    @pytest.mark.asyncio
    async def test_workflow_conditional_operators(self):
        workflow_yaml = """
name: OperatorTest
data_types:
  input:
    type: object
    properties:
      val:
        type: integer
  output:
    type: object
    properties:
      result:
        type: string
tasks:
  - name: GTE Task
    type: combine
    when:
      gte: [ { type: context, value: input.val }, 10 ]
    inputs:
      result:
        type: literal
        value: "GTE 10"
    output: output
  - name: LT Task
    type: combine
    when:
      lt: [ { type: context, value: input.val }, 5 ]
    inputs:
      result:
        type: literal
        value: "LT 5"
    output: output
  - name: NotEQ Task
    type: combine
    when:
      not_eq: [ { type: context, value: input.val }, 7 ]
    inputs:
      result:
        type: literal
        value: "Not 7"
    output: output
"""
        workflow = Workflow.from_yaml(workflow_yaml)

        # val=12 -> GTE runs, NotEQ runs. Result: "Not 7"
        workflow = Workflow.from_yaml(workflow_yaml)
        result1 = await workflow.run({"val": 12})
        assert result1.data.result == "Not 7"

        # val=7 -> GTE (false), LT (false), NotEQ (false). Result: None
        workflow = Workflow.from_yaml(workflow_yaml)
        result3 = await workflow.run({"val": 7})
        assert result3.data is None

        # val=10 -> GTE runs, NotEQ runs. Result: "Not 7"
        workflow = Workflow.from_yaml(workflow_yaml)
        result4 = await workflow.run({"val": 10})
        assert result4.data.result == "Not 7"

    @pytest.mark.asyncio
    async def test_workflow_conditional_is_true_and_len(self):
        workflow_yaml = """
name: TrueLenTest
data_types:
  input:
    type: object
    properties:
      is_active:
        type: boolean
      tags:
        type: array
        items:
          type: string
  output:
    type: object
    properties:
      status:
        type: string
tasks:
  - name: Active Check
    type: combine
    when:
      is_true: { type: context, value: input.is_active }
    inputs:
      status:
        type: literal
        value: "Active"
    output: output
  - name: Length Check
    type: combine
    when:
      len: [ { type: context, value: input.tags }, 3 ]
    inputs:
      status:
        type: literal
        value: "Has 3 tags"
    output: output
"""
        workflow = Workflow.from_yaml(workflow_yaml)

        # is_active=True, tags length 2 -> Result "Active"
        result1 = await workflow.run({"is_active": True, "tags": ["a", "b"]})
        assert result1.data.status == "Active"

        # is_active=False, tags length 3 -> Result "Has 3 tags"
        result2 = await workflow.run({"is_active": False, "tags": ["a", "b", "c"]})
        assert result2.data.status == "Has 3 tags"

        # Both match. Last one wins if they write to same output field.
        result3 = await workflow.run({"is_active": True, "tags": ["a", "b", "c"]})
        assert result3.data.status == "Has 3 tags"


class TestWorkflowHistory:
    @pytest.mark.asyncio
    async def test_workflow_load_from_history(self, agents_session_maker, monkeypatch):
        service = AgentService(agents_session_maker)

        # 1. Setup first run to populate history
        yaml_1 = """
name: Run1
description: First run
data_types:
  input:
    type: object
    properties:
      user_message: { type: string }
  output:
    type: object
    properties:
      search_results: { type: string }
tasks:
  - name: combine
    type: combine
    inputs:
      search_results: { type: literal, value: "initial results" }
    output: output
"""
        wf1 = Workflow.from_yaml(yaml_1, agent_service=service)

        res1 = await wf1.run(
            input_data={"user_message": "hello"}, external_id="session-123"
        )
        session_id = res1.session_id
        assert session_id is not None

        # 2. Setup second run to load from history
        yaml_2 = """
name: Run2
description: Second run
data_types:
  input:
    type: object
    properties:
      user_message: { type: string }
  output:
    type: object
    properties:
      prev_results: { type: string }
      current_msg: { type: string }
tasks:
  - name: load_and_combine
    type: combine
    inputs:
      prev_results: { type: history, value: search_results }
      current_msg: { type: context, value: input.user_message }
    output: output
"""
        wf2 = Workflow.from_yaml(yaml_2, agent_service=service)

        res2 = await wf2.run(input_data={"user_message": "bye"}, session_id=session_id)

        assert res2.data.prev_results == "initial results"
        assert res2.data.current_msg == "bye"

    @pytest.mark.asyncio
    async def test_workflow_condition_with_history(
        self, agents_session_maker, monkeypatch
    ):
        service = AgentService(agents_session_maker)

        # 1. Populate history
        yaml_1 = """
name: Run1
data_types:
  input: { type: object, properties: { msg: { type: string } } }
  output: { type: object, properties: { status: { type: string } } }
tasks:
  - name: t1
    type: combine
    inputs: { status: { type: literal, value: "completed" } }
    output: output
"""
        wf1 = Workflow.from_yaml(yaml_1, service)
        res1 = await wf1.run(input_data={"msg": "start"}, external_id="session-456")
        session_id = res1.session_id

        # 2. Run with condition checking history
        yaml_2 = """
name: Run2
data_types:
  input: { type: object, properties: { msg: { type: string } } }
  output: { type: object, properties: { result: { type: string } } }
tasks:
  - name: conditional_task
    type: combine
    when:
      eq:
        - { type: history, value: status }
        - "completed"
    inputs:
      result: { type: literal, value: "it worked" }
    output: output
"""
        wf2 = Workflow.from_yaml(yaml_2, service)
        res2 = await wf2.run(input_data={"msg": "next"}, session_id=session_id)

        assert res2.data is not None
        assert res2.data.result == "it worked"

        # 3. Run with condition that should fail
        yaml_3 = """
name: Run3
data_types:
  input: { type: object, properties: { msg: { type: string } } }
  output: { type: object, properties: { result: { type: string } } }
tasks:
  - name: conditional_task
    type: combine
    when:
      eq:
        - { type: history, value: status }
        - "failed"
    inputs:
      result: { type: literal, value: "should not happen" }
    output: output
"""
        wf3 = Workflow.from_yaml(yaml_3, service)
        res3 = await wf3.run(input_data={"msg": "next"}, session_id=session_id)
        assert res3.data is None


class TestWorkflowStop:
    @pytest.mark.asyncio
    async def test_workflow_stop_field(self):
        workflow_yaml = """
name: StopTest
data_types:
  input:
    type: object
    properties:
      should_stop:
        type: boolean
  output:
    type: object
    properties:
      task1_run:
        type: boolean
        required: false
        default: false
      task2_run:
        type: boolean
        required: false
        default: false
tasks:
  - name: Task 1
    type: combine
    stop: true
    inputs:
      task1_run:
        type: literal
        value: true
    output: output
  - name: Task 2
    type: combine
    inputs:
      task2_run:
        type: literal
        value: true
    output: output
"""
        workflow = Workflow.from_yaml(workflow_yaml)

        # Task 1 has stop: true, so Task 2 should NOT run
        result = await workflow.run({"should_stop": True})

        assert result.data is not None
        assert result.data.task1_run is True
        assert result.data.task2_run is False

    @pytest.mark.asyncio
    async def test_workflow_stop_with_when(self):
        workflow_yaml = """
name: StopWhenTest
data_types:
  input:
    type: object
    properties:
      stop_early:
        type: boolean
  output:
    type: object
    properties:
      step:
        type: string
tasks:
  - name: Step 1
    type: combine
    when:
      eq: [ { type: context, value: input.stop_early }, true ]
    stop: true
    output: output
    inputs:
      step:
        type: literal
        value: "stopped at step 1"
  - name: Step 2
    type: combine
    output: output
    inputs:
      step:
        type: literal
        value: "reached step 2"
"""
        workflow = Workflow.from_yaml(workflow_yaml)

        # Case 1: stop_early is true. Step 1 runs and stops the workflow.
        result1 = await workflow.run({"stop_early": True})
        assert result1.data.step == "stopped at step 1"

        # Case 2: stop_early is false. Step 1 is skipped, so stop: true is not triggered.
        # Step 2 should run.
        result2 = await workflow.run({"stop_early": False})
        assert result2.data.step == "reached step 2"


class TestWorkflowStreaming:
    @pytest.mark.asyncio
    async def test_workflow_streaming_prompt(self, monkeypatch):
        # 1. Setup Workflow
        workflow_yaml = """
name: StreamTest
description: Test streaming
data_types:
  input:
    type: object
    properties:
      user_message:
        type: string
  output:
    type: object
    properties:
      agent_response:
        type: string
tasks:
  - name: generate
    type: llm
    prompt: "Hello"
    inputs:
      user_message:
        type: context
        value: input.user_message
    output: output
    stream_updates: true
    stream_output: true
"""
        workflow = Workflow.from_yaml(workflow_yaml)

        # 2. Mock LLMClient.chat_completions to simulate streaming
        async def mock_chat_completions(
            self, response_model, messages, streamer=None, **kwargs
        ):
            if streamer:
                # Simulate partial stream
                await streamer.stream_partial('{"agent_response": "He')
                await streamer.stream_partial('{"agent_response": "Hello world"}')
                await streamer.stream_complete('{"agent_response": "Hello world"}')

            response = response_model(agent_response="Hello world")
            stats = None  # Not needed for this test
            return response, stats

        monkeypatch.setattr(LLMClient, "chat_completions", mock_chat_completions)

        # 3. Run Workflow with stream
        queue = asyncio.Queue()
        input_data = {"user_message": "Junie"}
        task = asyncio.create_task(workflow.run(input_data=input_data, queue=queue))

        # 4. Verify Stream Content
        lines = []
        while not task.done() or not queue.empty():
            try:
                line = await asyncio.wait_for(queue.get(), timeout=0.1)
                lines.append(line)
            except asyncio.TimeoutError:
                continue

        _ = await task

        assert len(lines) == 4

        running_task = StreamContent.model_validate_json(lines[0])
        assert running_task.type == "complete"
        assert running_task.name == "running_task"
        assert running_task.value == "generate"

        partial1 = StreamContent.model_validate_json(lines[1])
        assert partial1.type == "partial"
        assert partial1.name == "output"
        assert partial1.value == '{"agent_response": "He'

        partial2 = StreamContent.model_validate_json(lines[2])
        assert partial2.type == "partial"
        assert partial2.name == "output"
        assert partial2.value == '{"agent_response": "Hello world"}'

        complete_llm = StreamContent.model_validate_json(lines[3])
        assert complete_llm.type == "complete"
        assert complete_llm.name == "output"
        assert complete_llm.value == '{"agent_response": "Hello world"}'

    @pytest.mark.asyncio
    async def test_workflow_streaming_tool(self, monkeypatch):
        # 1. Setup Workflow with tool
        workflow_yaml = """
name: ToolStreamTest
description: Test tool streaming
rest_servers:
  - name: mock
    url: "http://mock"
data_types:
  input:
    type: object
    properties:
      user_message:
        type: string
  output:
    type: object
    properties:
      agent_response:
        type: string
tasks:
  - name: tool_call
    type: rest
    tool: "test"
    rest_server: mock
    inputs:
      msg:
        type: context
        value: input.user_message
    output: output
    stream_updates: true
    stream_output: true
"""
        workflow = Workflow.from_yaml(workflow_yaml)

        # 2. Mock httpx.AsyncClient.request
        class MockResponse:
            def __init__(self, json_data):
                self.json_data = json_data

            def json(self):
                return self.json_data

            def raise_for_status(self):
                pass

        async def mock_request(*args, **kwargs):
            return MockResponse({"agent_response": "Tool response"})

        monkeypatch.setattr("httpx.AsyncClient.request", mock_request)

        # 3. Run Workflow with stream
        queue = asyncio.Queue()
        input_data = {"user_message": "Junie"}
        task = asyncio.create_task(workflow.run(input_data=input_data, queue=queue))

        # 4. Verify Stream Content
        lines = []
        while not task.done() or not queue.empty():
            try:
                line = await asyncio.wait_for(queue.get(), timeout=0.1)
                lines.append(line)
            except asyncio.TimeoutError:
                continue

        _ = await task

        # 4. Verify Stream Content
        assert len(lines) == 1

        running_task = StreamContent.model_validate_json(lines[0])
        assert running_task.type == "complete"
        assert running_task.name == "running_task"
        assert running_task.value == "tool_call"

    @pytest.mark.asyncio
    async def test_workflow_python_function_registration(self, monkeypatch):
        # 1. Setup Workflow with python_function
        # Use a simple function for testing
        def my_test_concat(a: str, b: str) -> str:
            return a + b

        import sys
        import types

        m = types.ModuleType("my_mock_module")
        m.my_test_concat = my_test_concat
        sys.modules["my_mock_module"] = m

        workflow_yaml = """
name: PythonFuncRegTest
description: Test python function registration
python_functions:
  - name: "my_test_func"
    path: "my_mock_module.my_test_concat"
data_types:
  input:
    type: object
    properties:
      p1:
        type: string
      p2:
        type: string
  output:
    type: object
    properties:
      result:
        type: string
tasks:
  - name: call_python
    type: python
    python_tool: "my_test_func"
    inputs:
      a:
        type: context
        value: input.p1
      b:
        type: context
        value: input.p2
    output: output
"""
        workflow = Workflow.from_yaml(workflow_yaml)

        # 2. Run Workflow
        input_data = {"p1": "hello ", "p2": "world"}
        result = await workflow.run(input_data=input_data)

        # 3. Verify Result
        assert result.data.result == "hello world"

    @pytest.mark.asyncio
    async def test_workflow_concurrent_runs(self, agents_session_maker, monkeypatch):
        # 1. Setup simple workflow
        workflow_yaml = """
name: ConcurrentTest
data_types:
  input:
    type: object
    properties:
      user_message:
        type: string
  output:
    type: object
    properties:
      agent_response:
        type: string
tasks:
  - name: task1
    type: llm
    prompt: "Say hello to user"
    inputs:
      user:
        type: context
        value: input.user_message
    output: output
"""
        workflow = Workflow.from_yaml(workflow_yaml)

        # 2. Mock chat_completions with a delay
        async def mock_chat_completions(
            self_client, response_model, messages, **kwargs
        ):
            await asyncio.sleep(0.1)  # Simulate some processing time
            # The system message is messages[0], its content is the prompt
            prompt_content = messages[0]["content"]

            # Since the prompt is constructed as prompt + "\nINPUT DATA:" + ...
            # we check for our values in the prompt_content
            user_msg = "Unknown"
            if "Alice" in prompt_content:
                user_msg = "Alice"
            elif "Bob" in prompt_content:
                user_msg = "Bob"

            response = response_model(agent_response=f"Hello {user_msg}")
            from kavalai.agents.db import ModelCallStat

            stats = ModelCallStat(call_type="llm", model=self_client.full_model)
            return response, stats

        monkeypatch.setattr(
            "kavalai.llm_clients.llm_client.LLMClient.chat_completions",
            mock_chat_completions,
        )

        # 3. Define a runner function that creates its own session and service
        async def run_one(user_message):
            agent_service = AgentService(agents_session_maker)
            # Create a new workflow instance with the agent_service
            wf = Workflow(workflow.workflow_model, agent_service=agent_service)
            return await wf.run(
                input_data={"user_message": user_message},
            )

        # 4. Run two instances concurrently
        results = await asyncio.gather(run_one("Alice"), run_one("Bob"))

        # 5. Verify results
        assert results[0].data.agent_response.startswith("Hello")
        assert "Alice" in results[0].data.agent_response
        assert results[1].data.agent_response.startswith("Hello")
        assert "Bob" in results[1].data.agent_response

        # Check that sessions are different (they should be as we created them separately)
        assert results[0].session_id != results[1].session_id


from kavalai.agents.planning_agent import get_step_output_type


class TestWorkflowPlanningAgent:
    class MockOutput(BaseModel):
        answer: str

    @pytest.mark.asyncio
    async def test_run_planning_agent_success(self):
        # 1. Setup Workflow Model with an AgentTask
        workflow_data = {
            "name": "test_workflow",
            "llm_model": "openai/test-model",
            "data_types": {
                "input": {
                    "type": "object",
                    "properties": {"user_message": {"type": "string"}},
                },
                "output": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                },
            },
            "tasks": [
                {
                    "name": "planner_task",
                    "type": "agent",
                    "prompt": "Plan and solve this: {{input.user_message}}",
                    "output": "output",
                    "max_steps": 3,
                    "inputs": {
                        "user_msg": {"type": "context", "value": "input.user_message"}
                    },
                }
            ],
        }
        workflow_model = WorkflowModel(**workflow_data)
        workflow = Workflow(workflow_model)

        # 2. Setup RunContext and Task
        run_context = RunContext()
        run_context.data["input"] = workflow.get_data_type("input")(
            user_message="hello"
        )
        task = workflow_model.tasks[0]

        # 3. Mock LLMClient and its response
        StepOutput = get_step_output_type(self.MockOutput)
        final_output = self.MockOutput(answer="Final result")
        step_output = StepOutput(
            short_explanation="Done",
            instructions="Proceed",
            tool_calls=[],
            output=final_output,
        )

        with patch("kavalai.agents.workflow.LLMClient") as MockLLMClient:
            mock_client_instance = MockLLMClient.return_value
            mock_client_instance.chat_completions = AsyncMock(
                return_value=(step_output, {})
            )

            # 4. Run the planning agent task
            # Set up mock agent service to verify add_task call
            mock_agent_service = AsyncMock(spec=AgentService)
            workflow.agent_service = mock_agent_service
            workflow.task_logger.agent_service = mock_agent_service
            # Need to update workflow's run_context because task_logger uses it
            workflow.run_context.run_id = uuid4()
            workflow.run_context.agent_id = uuid4()
            workflow.run_context.session_id = uuid4()
            # Also update the run_context being passed for consistency
            run_context.run_id = workflow.run_context.run_id
            run_context.agent_id = workflow.run_context.agent_id
            run_context.session_id = workflow.run_context.session_id

            await workflow.run_agent_task(task, run_context, None)

            # 5. Assertions
            assert run_context.data["planner_task"] == final_output
            assert run_context.data["output"] == final_output

            # Verify add_task was called for both the step and the overall task
            assert mock_agent_service.add_task.call_count == 2

            # Check the first call (step 0)
            args0, kwargs0 = mock_agent_service.add_task.call_args_list[0]
            assert kwargs0["name"] == "planner_task_step_0"

            # Check the second call (overall task)
            args1, kwargs1 = mock_agent_service.add_task.call_args_list[1]
            assert kwargs1["name"] == "planner_task"
            assert kwargs1["output"] == to_plain(final_output)

            # Verify LLMClient was initialized with the correct model
            MockLLMClient.assert_called_once_with(model="openai/test-model")

            # Verify chat_completions was called
            assert mock_client_instance.chat_completions.called
            args, kwargs = mock_client_instance.chat_completions.call_args
            # We can't easily compare classes created by get_step_output_type because they are different at each call
            # but we can check if it's a StepOutput by its name
            assert kwargs["response_model"].__name__ == "StepOutput"

    @pytest.mark.asyncio
    async def test_run_planning_agent_streaming(self):
        # 1. Setup Workflow Model with an AgentTask and streaming enabled
        workflow_data = {
            "name": "test_workflow",
            "llm_model": "openai/test-model",
            "data_types": {
                "output": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                },
            },
            "tasks": [
                {
                    "name": "planner_task",
                    "type": "agent",
                    "prompt": "Plan and solve this",
                    "output": "output",
                    "max_steps": 3,
                    "stream_updates": True,
                    "stream_output": True,
                }
            ],
        }
        workflow_model = WorkflowModel(**workflow_data)
        workflow = Workflow(workflow_model)

        # 2. Setup RunContext and Queue
        run_context = RunContext()
        queue = asyncio.Queue()
        task = workflow_model.tasks[0]

        # 3. Mock LLMClient and its response
        StepOutput = get_step_output_type(self.MockOutput)
        final_output = self.MockOutput(answer="Final result")
        step_output = StepOutput(
            short_explanation="Iteration 1",
            instructions="Proceed",
            tool_calls=[],
            output=final_output,
        )

        with patch("kavalai.agents.workflow.LLMClient") as MockLLMClient:
            mock_client_instance = MockLLMClient.return_value
            mock_client_instance.chat_completions = AsyncMock(
                return_value=(step_output, {})
            )

            # 4. Run the planning agent task
            await workflow.run_agent_task(task, run_context, queue)

            # 5. Check queue for streamed updates and output
            streamed_items = []
            while not queue.empty():
                item_json = await queue.get()
                streamed_items.append(json.loads(item_json))

            # Expecting:
            # 1. Update from loop in workflow.py (task name) if we ran via run()
            # but we called run_planning_agent directly, so:
            # - Iteration 1 (from PlanningAgent.run)
            # - Final result (from PlanningAgent.run)

            # Since we call run_planning_agent directly, it doesn't have the task loop from run()

            assert any(
                item["name"] == "running_task" and item["value"] == "Iteration 1"
                for item in streamed_items
            )
            assert any(
                item["name"] == "output"
                and item["value"] == final_output.model_dump_json()
                for item in streamed_items
            )

    @pytest.mark.asyncio
    async def test_run_planning_agent_no_streaming(self):
        # 1. Setup Workflow Model with an AgentTask and streaming disabled
        workflow_data = {
            "name": "test_workflow",
            "llm_model": "openai/test-model",
            "data_types": {
                "output": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                },
            },
            "tasks": [
                {
                    "name": "planner_task",
                    "type": "agent",
                    "prompt": "Plan and solve this",
                    "output": "output",
                    "max_steps": 3,
                    "stream_updates": False,
                    "stream_output": False,
                }
            ],
        }
        workflow_model = WorkflowModel(**workflow_data)
        workflow = Workflow(workflow_model)

        # 2. Setup RunContext and Queue
        run_context = RunContext()
        queue = asyncio.Queue()
        task = workflow_model.tasks[0]

        # 3. Mock LLMClient and its response
        StepOutput = get_step_output_type(self.MockOutput)
        final_output = self.MockOutput(answer="Final result")
        step_output = StepOutput(
            short_explanation="Iteration 1",
            instructions="Proceed",
            tool_calls=[],
            output=final_output,
        )

        with patch("kavalai.agents.workflow.LLMClient") as MockLLMClient:
            mock_client_instance = MockLLMClient.return_value
            mock_client_instance.chat_completions = AsyncMock(
                return_value=(step_output, {})
            )

            # 4. Run the planning agent task
            await workflow.run_agent_task(task, run_context, queue)

            # 5. Check queue (should be empty because streaming is disabled for this task)
            assert queue.empty()

    @pytest.mark.asyncio
    async def test_run_planning_agent_stream_persisted(self):
        # 1. Setup Workflow Model with an AgentTask and stream_persisted enabled
        workflow_data = {
            "name": "test_workflow",
            "llm_model": "openai/test-model",
            "data_types": {
                "output": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                },
            },
            "tasks": [
                {
                    "name": "planner_task",
                    "type": "agent",
                    "prompt": "Plan and solve this",
                    "output": "output",
                    "max_steps": 3,
                    "stream_persisted": True,
                    "stream_output": True,
                }
            ],
        }
        workflow_model = WorkflowModel(**workflow_data)
        workflow = Workflow(workflow_model)

        # 2. Setup RunContext and Queue
        run_context = RunContext()
        queue = asyncio.Queue()
        task = workflow_model.tasks[0]

        # 3. Mock LLMClient and its response
        from kavalai.agents.planning_agent import get_step_output_type, ToolCall

        StepOutput = get_step_output_type(self.MockOutput)

        tool_call = ToolCall(
            name="test_tool",
            literal_args=json.dumps({"arg": "val"}),
            persist_to="persisted_val",
            call_id="call_1",
        )

        step_output_1 = StepOutput(
            short_explanation="Iteration 1",
            instructions="Calling tool",
            tool_calls=[tool_call],
            output=None,
        )

        final_output = self.MockOutput(answer="Final result")
        step_output_2 = StepOutput(
            short_explanation="Iteration 2",
            instructions="Done",
            tool_calls=[],
            output=final_output,
        )

        with patch("kavalai.agents.workflow.LLMClient") as MockLLMClient:
            mock_client_instance = MockLLMClient.return_value
            mock_client_instance.chat_completions = AsyncMock(
                side_effect=[(step_output_1, {}), (step_output_2, {})]
            )

            # Mock Kernel call_tool
            workflow.kernel.call_tool = AsyncMock(return_value="tool_result_value")

            # 4. Run the planning agent task
            await workflow.run_agent_task(task, run_context, queue)

            # 5. Check queue for streamed persisted value
            streamed_items = []
            while not queue.empty():
                item_json = await queue.get()
                streamed_items.append(json.loads(item_json))

            # Expecting:
            # - persisted_val (from PlanningAgent.run because stream_persisted is True)
            # - output (from PlanningAgent.run because stream_output is True)

            assert any(
                item["name"] == "persisted_val" and item["value"] == "tool_result_value"
                for item in streamed_items
            )
            assert any(
                item["name"] == "output"
                and item["value"] == final_output.model_dump_json()
                for item in streamed_items
            )

    @pytest.mark.asyncio
    async def test_run_planning_agent_with_history(self):
        # 1. Setup Workflow Model
        workflow_data = {
            "name": "test_workflow_history",
            "llm_model": "openai/test-model",
            "data_types": {
                "input": {
                    "type": "object",
                    "properties": {"user_message": {"type": "string"}},
                },
                "output": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                },
            },
            "tasks": [
                {
                    "name": "planner_task",
                    "type": "agent",
                    "prompt": "Solve",
                    "output": "output",
                    "use_history": True,
                    "inputs": {},
                }
            ],
        }
        workflow_model = WorkflowModel(**workflow_data)

        # Mock AgentService
        mock_agent_service = MagicMock()
        mock_history_msg = MagicMock()
        mock_history_msg.role = "user"
        mock_history_msg.content = "previous message"
        mock_agent_service.get_chat_history = AsyncMock(return_value=[mock_history_msg])

        workflow = Workflow(workflow_model, agent_service=mock_agent_service)

        # 2. Setup RunContext with session_id
        import uuid

        session_id = uuid.uuid4()
        run_context = RunContext(session_id=session_id)
        run_context.data["input"] = workflow.get_data_type("input")(
            user_message="hello"
        )
        task = workflow_model.tasks[0]

        # 3. Mock PlanningAgent to avoid deep nesting of mocks
        with patch("kavalai.agents.workflow.PlanningAgent") as MockPlanningAgent:
            mock_planner_instance = MockPlanningAgent.return_value
            mock_planner_instance.run = AsyncMock(
                return_value=self.MockOutput(answer="History result")
            )

            # 4. Run
            await workflow.run_agent_task(task, run_context, None)

            # 5. Verify history was fetched and passed
            mock_agent_service.get_chat_history.assert_awaited_once_with(session_id)

            # Verify PlanningAgent was initialized and run with history
            MockPlanningAgent.assert_called_once()
            mock_planner_instance.run.assert_awaited_once()
            run_args, run_kwargs = mock_planner_instance.run.call_args
            assert run_kwargs["chat_history"] == [
                {"role": "user", "content": "previous message"}
            ]
