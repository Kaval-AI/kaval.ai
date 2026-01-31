import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock
from sqlalchemy.ext.asyncio import async_sessionmaker
from kavalai.agents.server import create_agent_app
from kavalai.agents.workflow import Workflow, WorkflowModel


@pytest.fixture
def mock_workflow():
    model = WorkflowModel(
        name="test_agent",
        description="Test description",
        version="1.0.0",
        data_types={
            "input": {"type": "object", "properties": {"query": {"type": "string"}}},
            "output": {"type": "object", "properties": {"result": {"type": "string"}}},
        },
        tasks=[],
    )
    workflow = Workflow(model)
    return workflow


def test_liveness_endpoint(mock_workflow):
    app = create_agent_app(mock_workflow)
    client = TestClient(app)
    response = client.get("/liveness")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_health_endpoint_success(mock_workflow):
    mock_session = AsyncMock()
    mock_session.execute.return_value = None

    # We need a context manager that returns our mock_session
    class MockAsyncContextManager:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, exc_type, exc, tb):
            pass

    mock_session_provider = MagicMock(spec=async_sessionmaker)
    mock_session_provider.return_value = MockAsyncContextManager()

    app = create_agent_app(mock_workflow, session_provider=mock_session_provider)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "connected"}

    # Verify DB was called
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_health_endpoint_failure(mock_workflow):
    # Mocking failure
    class MockAsyncContextManager:
        async def __aenter__(self):
            raise Exception("DB Error")

        async def __aexit__(self, exc_type, exc, tb):
            pass

    mock_session_provider = MagicMock(spec=async_sessionmaker)
    mock_session_provider.return_value = MockAsyncContextManager()

    app = create_agent_app(mock_workflow, session_provider=mock_session_provider)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["detail"] == "Database connection failed"
