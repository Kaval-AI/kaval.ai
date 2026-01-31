"""Tests for workflow.py REST server environment variable handling."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from pydantic import BaseModel
from kavalai.agents.workflow import (
    Workflow,
    WorkflowModel,
    WorkflowException,
    RestServer,
    Task,
    TypeInputInfo,
    RunContext,
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
            await workflow.run_tool(task, run_context)

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
            await workflow.run_tool(task, run_context)

        assert captured_method == "POST"
        # For POST, it should use json
        # Since only one input is given, it is passed straight
        assert captured_json == run_context.data["input"]
        # When params is popped, it's missing from kwargs in request call
        assert captured_params is None

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
            await workflow.run_tool(workflow.workflow_model.tasks[0], run_context)

        # Check that the Pydantic model was converted to a dict
        # Since only one input is given, it is passed straight
        assert captured_json == {"name": "test_value"}
        assert isinstance(captured_json, dict)


class TestRestServerEnvVarValidation:
    """Tests for REST server environment variable validation during workflow loading."""

    def test_workflow_loads_without_auth_env_vars(self):
        """Workflow should load successfully when no auth env vars are specified."""
        model = create_workflow_model_with_rest_server()
        workflow = Workflow(model)
        assert workflow is not None
        assert "test_server" in workflow.rest_servers

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
            await workflow.run_tool(task, run_context)

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
            await workflow.run_tool(task, run_context)

        assert captured_auth is None
