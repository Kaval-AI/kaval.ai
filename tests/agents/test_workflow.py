"""Tests for workflow.py REST server environment variable handling."""

from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, ValidationError

from kavalai.agents.workflow import Workflow
from kavalai.agents.run_context import RunContext
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
    async def test_run_tool_uses_default_get_method(self):
        """run_tool should use GET method by default."""
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
            await workflow.run_tool(task, run_context, None)

        assert captured_method == "GET"
        assert captured_url == "http://localhost:8000/test_tool"

    @pytest.mark.asyncio
    async def test_run_tool_uses_specified_post_method(self):
        """run_tool should use POST method when specified."""
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
            await workflow.run_tool(task, run_context, None)

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
    async def test_run_tool_serializes_pydantic_models(self):
        """run_tool should serialize Pydantic models to dicts in JSON body."""

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
            await workflow.run_tool(workflow.workflow_model.tasks[0], run_context, None)

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


class TestRunToolAuth:
    """Tests for run_tool method authentication handling."""

    @pytest.mark.asyncio
    async def test_run_tool_uses_basic_auth_when_env_vars_defined(self, monkeypatch):
        """run_tool should pass basic auth to AsyncClient when env vars are defined."""
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
            await workflow.run_tool(task, run_context, None)

        assert captured_auth == ("user123", "pass456")

    @pytest.mark.asyncio
    async def test_run_tool_no_auth_when_env_vars_not_defined(self, monkeypatch):
        """run_tool should not pass auth to AsyncClient when no env vars are defined."""
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
            await workflow.run_tool(task, run_context, None)

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
        from kavalai.agents.workflow import Workflow

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
