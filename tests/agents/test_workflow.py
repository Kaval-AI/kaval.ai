"""Tests for workflow.py REST server environment variable handling."""

from unittest.mock import patch, AsyncMock, MagicMock
import asyncio

import pytest
from pydantic import BaseModel, ValidationError

from kavalai.agents.workflow import Workflow
from kavalai.agents.run_context import RunContext
from kavalai.agents.agent_service import AgentService
from kavalai.llm_clients.common import StreamContent
from kavalai.agents.workflow_model import (
    WorkflowModel,
    WorkflowException,
    RestServer,
    Task,
    TypeInputInfo,
)


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
            Task(
                name="test_task",
                inputs={"query": TypeInputInfo(type="context", name="input")},
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

        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        captured_method = None
        captured_url = None

        async def mock_request(method, url, **kwargs):
            nonlocal captured_method, captured_url
            captured_method = method
            captured_url = url
            return mock_response

        mock_client.request = mock_request

        with patch(
            "kavalai.agents.workflow.httpx.AsyncClient", return_value=mock_client
        ):
            task = workflow.workflow_model.tasks[0]
            run_context = RunContext()
            run_context.data["input"] = MagicMock()
            await workflow.run_rest_tool(task, run_context, None)

        assert captured_method == "GET"
        assert captured_url == "http://localhost:8000/test_tool"

    @pytest.mark.asyncio
    async def test_run_rest_tool_uses_specified_post_method(self):
        """run_rest_tool should use POST method when specified."""
        model = create_workflow_model_with_rest_server(method="post")
        workflow = Workflow(model)

        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        captured_method = None
        captured_params = None
        captured_json = None

        async def mock_request(method, url, **kwargs):
            nonlocal captured_method, captured_params, captured_json
            captured_method = method
            captured_params = kwargs.get("params")
            captured_json = kwargs.get("json")
            return mock_response

        mock_client.request = mock_request

        with patch(
            "kavalai.agents.workflow.httpx.AsyncClient", return_value=mock_client
        ):
            task = workflow.workflow_model.tasks[0]
            run_context = RunContext()
            run_context.data["input"] = MagicMock()
            await workflow.run_rest_tool(task, run_context, None)

        assert captured_method == "POST"
        # For POST, it should use json
        # After changing prepare_tool_inputs to always return a dict,
        # single inputs are now wrapped in a dict with the input name as key.
        assert captured_json == {"query": run_context.data["input"]}
        # When params is popped, it's missing from kwargs in request call
        assert captured_params is None


class TestWorkflowTemperatureValidation:
    """Tests for temperature validation in Workflow."""

    def test_workflow_invalid_temperature_too_high(self):
        """Workflow should raise exception if temperature > 2.0."""
        data = {
            "name": "test",
            "temperature": 2.1,
            "data_types": {
                "input": {"type": "object", "properties": {}},
                "output": {"type": "object", "properties": {}},
            },
            "tasks": [],
        }
        model = WorkflowModel(**data)
        with pytest.raises(
            WorkflowException, match="temperature must be between 0.0 and 2.0"
        ):
            Workflow(model)

    def test_workflow_invalid_temperature_too_low(self):
        """Workflow should raise exception if temperature < 0.0."""
        data = {
            "name": "test",
            "temperature": -0.1,
            "data_types": {
                "input": {"type": "object", "properties": {}},
                "output": {"type": "object", "properties": {}},
            },
            "tasks": [],
        }
        model = WorkflowModel(**data)
        with pytest.raises(
            WorkflowException, match="temperature must be between 0.0 and 2.0"
        ):
            Workflow(model)

    def test_task_invalid_temperature(self):
        """Task should raise exception if temperature is invalid."""
        data = {
            "name": "test",
            "data_types": {
                "input": {"type": "object", "properties": {}},
                "output": {"type": "object", "properties": {}},
            },
            "tasks": [{"name": "task1", "temperature": 2.5, "output": "output"}],
        }
        model = WorkflowModel(**data)
        with pytest.raises(
            WorkflowException, match="task1' temperature must be between 0.0 and 2.0"
        ):
            Workflow(model)

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
                Task(
                    name="Task 1",
                    tool="search",
                    rest_server="test_server",
                    method="post",
                    inputs={"input": TypeInputInfo(type="context", value="input")},
                    output="output_model",
                )
            ],
        )
        workflow = Workflow(model)

        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        captured_json = None

        async def mock_request(method, url, **kwargs):
            nonlocal captured_json
            captured_json = kwargs.get("json")
            return mock_response

        mock_client.request = mock_request

        with patch(
            "kavalai.agents.workflow.httpx.AsyncClient", return_value=mock_client
        ):
            run_context = RunContext()
            run_context.data["input"] = TestModel(name="test_value")
            await workflow.run_rest_tool(
                workflow.workflow_model.tasks[0], run_context, None
            )

        # Check that the Pydantic model was converted to a dict
        # After changing prepare_tool_inputs to always return a dict,
        # single inputs are now wrapped in a dict with the input name as key.
        assert captured_json == {"input": {"name": "test_value"}}
        assert isinstance(captured_json, dict)


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
        assert "test_server" in workflow.rest_servers

    def test_workflow_loads_with_url_env(self, monkeypatch):
        monkeypatch.setenv("MY_REST_URL", "http://my-api.com")
        # create_workflow_model_with_rest_server uses url by default
        # We need to create a model where only url_env is specified
        rest_servers = [RestServer(name="test_server", url_env="MY_REST_URL")]
        model = create_workflow_model_with_rest_server()
        model.rest_servers = rest_servers

        workflow = Workflow(model)
        # url should stay None
        assert workflow.rest_servers["test_server"].url is None
        # url_env should stay as specified
        assert workflow.rest_servers["test_server"].url_env == "MY_REST_URL"

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
        """run_rest_tool should pass basic auth to AsyncClient when env vars are defined."""
        monkeypatch.setenv("TEST_USERNAME", "user123")
        monkeypatch.setenv("TEST_PASSWORD", "pass456")

        model = create_workflow_model_with_rest_server(
            username_env="TEST_USERNAME", password_env="TEST_PASSWORD"
        )
        workflow = Workflow(model)

        # Mock httpx.AsyncClient
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        captured_auth = None

        def capture_auth(*args, **kwargs):
            nonlocal captured_auth
            captured_auth = kwargs.get("auth")
            return mock_client

        with patch("kavalai.agents.workflow.httpx.AsyncClient", capture_auth):
            task = workflow.workflow_model.tasks[0]
            run_context = RunContext()
            run_context.data["input"] = MagicMock()
            await workflow.run_rest_tool(task, run_context, None)

        assert captured_auth == ("user123", "pass456")

    @pytest.mark.asyncio
    async def test_run_rest_tool_no_auth_when_env_vars_not_defined(self, monkeypatch):
        """run_rest_tool should not pass auth to AsyncClient when no env vars are defined."""
        model = create_workflow_model_with_rest_server()
        workflow = Workflow(model)

        # Mock httpx.AsyncClient
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        captured_auth = "NOT_SET"  # Use sentinel to distinguish from None

        def capture_auth(*args, **kwargs):
            nonlocal captured_auth
            captured_auth = kwargs.get("auth")
            return mock_client

        with patch("kavalai.agents.workflow.httpx.AsyncClient", capture_auth):
            task = workflow.workflow_model.tasks[0]
            run_context = RunContext()
            run_context.data["input"] = MagicMock()
            await workflow.run_rest_tool(task, run_context, None)

        assert captured_auth is None


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
    output:
      final_msg: { value: input.msg, type: context }
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
    inputs:
      text: { value: input.msg, type: context }
    output: intermediate
  - name: Final Output
    output:
      final_msg: { value: intermediate.text, type: context }
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
    when:
      eq: [ { type: context, value: input.run_task }, true ]
    inputs:
      val:
        type: context
        value: input.value
    output:
      result:
        type: literal
        value: "Task executed"
  - name: GT Task
    when:
      gt: [ { type: context, value: input.value }, 10 ]
    output:
      result:
        type: literal
        value: "Value is large"
  - name: All Task
    when:
      all:
        - eq: [ { type: context, value: input.run_task }, true ]
        - gt: [ { type: context, value: input.value }, 5 ]
    output:
      result:
        type: literal
        value: "All conditions met"
"""
        workflow = Workflow.from_yaml(workflow_yaml)

        # Case 1: run_task=False, value=5 -> No tasks should run, output should be empty/None
        result1 = await workflow.run({"run_task": False, "value": 5})
        assert result1.data is None

        # Case 2: run_task=True, value=5 -> Only "Conditional Task" should run
        result2 = await workflow.run({"run_task": True, "value": 5})
        assert result2.data.result == "Task executed"

        # Case 3: run_task=False, value=15 -> Only "GT Task" should run
        result3 = await workflow.run({"run_task": False, "value": 15})
        assert result3.data.result == "Value is large"

        # Case 4: run_task=True, value=15 -> All tasks run, last one wins for "result"
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
    when:
      any:
        - eq: [ { type: context, value: input.val }, 1 ]
        - eq: [ { type: context, value: input.val }, 2 ]
    output:
      result:
        type: literal
        value: "One or Two"
  - name: Not Task
    when:
      not:
        eq: [ { type: context, value: input.val }, 1 ]
    output:
      result:
        type: literal
        value: "Not One"
"""
        workflow = Workflow.from_yaml(workflow_yaml)

        # val=1 -> "Any Task" runs, "Not Task" does not. Result: "One or Two"
        result1 = await workflow.run({"val": 1})
        assert result1.data.result == "One or Two"

        # val=2 -> "Any Task" runs, "Not Task" runs. Result: "Not One"
        result2 = await workflow.run({"val": 2})
        assert result2.data.result == "Not One"

        # val=3 -> "Any Task" does not run, "Not Task" runs. Result: "Not One"
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
    when:
      gt: [ { type: context, value: "input.criteria.keywords.length" }, 0 ]
    output:
      result:
        type: literal
        value: "Found keywords"
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
    when:
      contains: [ { type: context, value: "input.tags" }, "vip" ]
    output:
      result:
        type: literal
        value: "VIP Customer"
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
    when:
      eq: [ { type: context, value: input.val }, 1 , 2]
    output:
      result:
        type: literal
        value: "Should fail"
"""
        # The error now happens during Workflow.from_yaml (initialization) due to Pydantic validator
        with pytest.raises(
            ValueError, match="Operator 'eq' requires a list of 2 operands."
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
    when:
      gte: [ { type: context, value: input.val }, 10 ]
    output:
      result:
        type: literal
        value: "GTE 10"
  - name: LT Task
    when:
      lt: [ { type: context, value: input.val }, 5 ]
    output:
      result:
        type: literal
        value: "LT 5"
  - name: NotEQ Task
    when:
      not_eq: [ { type: context, value: input.val }, 7 ]
    output:
      result:
        type: literal
        value: "Not 7"
"""
        workflow = Workflow.from_yaml(workflow_yaml)

        # val=12 -> GTE runs, NotEQ runs. Result: "Not 7"
        result1 = await workflow.run({"val": 12})
        assert result1.data.result == "Not 7"

        # val=7 -> GTE (false), LT (false), NotEQ (false). Result: None
        result3 = await workflow.run({"val": 7})
        assert result3.data is None

        # val=10 -> GTE runs, NotEQ runs. Result: "Not 7"
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
    when:
      is_true: { type: context, value: input.is_active }
    output:
      status:
        type: literal
        value: "Active"
  - name: Length Check
    when:
      len: [ { type: context, value: input.tags }, 3 ]
    output:
      status:
        type: literal
        value: "Has 3 tags"
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
    async def test_workflow_load_from_history(self, agents_db, monkeypatch):
        service = AgentService(agents_db)

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
    inputs:
      search_results: { type: literal, value: "initial results" }
    output:
      search_results: { type: literal, value: "initial results" }
"""
        wf1 = Workflow.from_yaml(yaml_1)
        wf1.agent_service = service

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
    inputs:
      prev_results: { type: history, value: search_results }
      current_msg: { type: context, value: input.user_message }
    output:
      prev_results: { type: history, value: search_results }
      current_msg: { type: context, value: input.user_message }
"""
        wf2 = Workflow.from_yaml(yaml_2)
        wf2.agent_service = service

        res2 = await wf2.run(input_data={"user_message": "bye"}, session_id=session_id)

        assert res2.data.prev_results == "initial results"
        assert res2.data.current_msg == "bye"

    @pytest.mark.asyncio
    async def test_workflow_condition_with_history(self, agents_db, monkeypatch):
        service = AgentService(agents_db)

        # 1. Populate history
        yaml_1 = """
name: Run1
data_types:
  input: { type: object, properties: { msg: { type: string } } }
  output: { type: object, properties: { status: { type: string } } }
tasks:
  - name: t1
    inputs: { status: { type: literal, value: "completed" } }
    output: { status: { type: literal, value: "completed" } }
"""
        wf1 = Workflow.from_yaml(yaml_1)
        wf1.agent_service = service
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
    when:
      eq:
        - { type: history, value: status }
        - "completed"
    inputs:
      result: { type: literal, value: "it worked" }
    output: { result: { type: literal, value: "it worked" } }
"""
        wf2 = Workflow.from_yaml(yaml_2)
        wf2.agent_service = service
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
    when:
      eq:
        - { type: history, value: status }
        - "failed"
    inputs:
      result: { type: literal, value: "should not happen" }
    output: { result: { type: literal, value: "should not happen" } }
"""
        wf3 = Workflow.from_yaml(yaml_3)
        wf3.agent_service = service
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
    stop: true
    output:
      task1_run:
        type: literal
        value: true
  - name: Task 2
    output:
      task2_run:
        type: literal
        value: true
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
    when:
      eq: [ { type: context, value: input.stop_early }, true ]
    stop: true
    output:
      step:
        type: literal
        value: "stopped at step 1"
  - name: Step 2
    output:
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
    prompt: "Hello {{input.user_message}}"
    output: output
    stream: true
"""
        workflow = Workflow.from_yaml(workflow_yaml)

        # 2. Mock chat_completions to simulate streaming
        async def mock_chat_completions(
            model, response_model, messages, streamer=None, **kwargs
        ):
            if streamer:
                # Simulate partial stream
                await streamer.stream_partial('{"agent_response": "He')
                await streamer.stream_partial('{"agent_response": "Hello world"}')
                await streamer.stream_complete('{"agent_response": "Hello world"}')

            response = response_model(agent_response="Hello world")
            stats = None  # Not needed for this test
            return response, stats

        monkeypatch.setattr(
            "kavalai.agents.workflow.chat_completions", mock_chat_completions
        )

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

        result = await task

        assert len(lines) == 3

        partial1 = StreamContent.model_validate_json(lines[0])
        assert partial1.type == "partial"
        assert partial1.name == "output"
        assert partial1.value == '{"agent_response": "He'

        partial2 = StreamContent.model_validate_json(lines[1])
        assert partial2.type == "partial"
        assert partial2.name == "output"
        assert partial2.value == '{"agent_response": "Hello world"}'

        complete = StreamContent.model_validate_json(lines[2])
        assert complete.type == "complete"
        assert complete.name == "output"
        assert complete.value == '{"agent_response": "Hello world"}'

        assert result.data.agent_response == "Hello world"

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
    tool: "test"
    rest_server: mock
    inputs:
      msg:
        type: context
        value: input.user_message
    output: output
    stream: true
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

        result = await task

        # 4. Verify Stream Content
        assert len(lines) >= 1

        complete = StreamContent.model_validate_json(lines[-1])
        assert complete.type == "complete"
        assert complete.name == "output"
        assert complete.value == '{"agent_response":"Tool response"}'

        assert result.data.agent_response == "Tool response"
