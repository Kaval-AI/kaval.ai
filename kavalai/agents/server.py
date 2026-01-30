"""Launch Kaval.AI agent REST server.

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

import logging
import secrets
from contextlib import asynccontextmanager
from typing import Annotated
from typing import Optional, Union
from uuid import UUID

import uvicorn
from environs import Env
from fastapi import Depends
from fastapi import HTTPException, status, FastAPI, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import async_sessionmaker

from kavalai.agents.agent_service import AgentService
from kavalai.agents.db import db_manager
from kavalai.agents.workflow import Workflow

logger = logging.getLogger(__name__)

security = HTTPBasic()

env = Env()
env.read_env()


def validate_auth(credentials: Annotated[HTTPBasicCredentials, Depends(security)]):
    """Validate HTTP Basic Authentication.

    Authentication is disabled if KAVALAI_AGENT_BASIC_AUTH_USER and
    KAVALAI_AGENT_BASIC_AUTH_USER_PASSWORD are not set in the environment.

    Args:
        credentials: The credentials provided in the request.

    Returns:
        True if authentication is successful or disabled.

    Raises:
        HTTPException: If authentication fails.
    """
    expected_username = env.str("KAVALAI_AGENT_BASIC_AUTH_USER", "")
    expected_password = env.str("KAVALAI_AGENT_BASIC_AUTH_USER_PASSWORD", "")

    # Basic auth is disabled
    if not expected_username and not expected_password:
        return True

    is_correct_username = secrets.compare_digest(
        credentials.username, expected_username
    )
    is_correct_password = secrets.compare_digest(
        credentials.password, expected_password
    )

    if is_correct_username and is_correct_password:
        return True

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username/password",
        headers={"WWW-Authenticate": "Basic"},
    )


@asynccontextmanager
async def session_scope(session_or_factory):
    """Provide a database session from either a sessionmaker or an existing session.

    This context manager ensures that if a factory is provided, a new session
    is created and closed properly. If an existing session is provided, it
    is used as-is.
    """
    if isinstance(session_or_factory, async_sessionmaker):
        async with session_or_factory() as session:
            yield session
    else:
        yield session_or_factory


def create_agent_app(
    workflow: Workflow,
    session_provider: Union[async_sessionmaker, None] = None,
) -> FastAPI:
    """Create a FastAPI application for a given workflow.

    The application dynamically generates input and output models based on the
    workflow's schema and provides endpoints to run the agent and retrieve
    its configuration.

    Args:
        workflow: The Workflow instance to serve.
        session_provider: An optional SQLAlchemy async_sessionmaker to provide
            database sessions for agent execution.

    Returns:
        A FastAPI application instance.
    """
    app = FastAPI(
        title=workflow.workflow_model.name,
        description=workflow.workflow_model.description,
        version=workflow.workflow_model.version,
    )

    InputDataType = workflow.get_data_type("input")
    OutputDataType = workflow.get_data_type("output")

    # Define the request body schema.
    # It wraps the workflow's input data and adds session management fields.
    class InputType(BaseModel):
        session_id: Optional[UUID] = None
        external_id: Optional[str] = None
        data: InputDataType

    # Define the response body schema.
    # It wraps the workflow's output data and includes the session ID.
    class OutputType(BaseModel):
        session_id: Optional[UUID]
        data: OutputDataType

    @app.post("/run_agent", response_model=OutputType)
    async def run_agent(
        input_data: InputType,
        credentials: Annotated[HTTPBasicCredentials, Depends(security)],
    ) -> OutputType:
        """Execute the agent workflow with the provided input.

        This endpoint:
        1. Validates authentication.
        2. Creates a database session.
        3. Initializes the AgentService.
        4. Runs the workflow with the provided data and session parameters.
        5. Returns the execution result and session ID.
        """
        validate_auth(credentials)
        async with session_scope(session_provider) as session:
            workflow.agent_service = AgentService(session)
            result = await workflow.run(
                input_data=input_data.data.model_dump(),
                session_id=input_data.session_id,
                external_id=input_data.external_id,
            )
            return OutputType(session_id=result.session_id, data=result.data)

    @app.get("/workflow", response_model=OutputType)
    async def get_workflow(
        credentials: Annotated[HTTPBasicCredentials, Depends(security)],
    ):
        """Retrieve the workflow configuration.

        Returns the full YAML-derived workflow model in JSON format.
        """
        validate_auth(credentials)
        return Response(
            content=workflow.workflow_model.model_dump_json(),
            media_type="application/json",
        )

    return app


def mask_db_uri(db_uri: str) -> str:
    """Mask the password in a database URI.

    Args:
        db_uri: The database URI to mask.

    Returns:
        The masked database URI.
    """
    if "@" not in db_uri:
        return db_uri

    try:
        prefix, rest = db_uri.split("://", 1)
        auth, host_path = rest.split("@", 1)
        if ":" in auth:
            user, password = auth.split(":", 1)
            return f"{prefix}://{user}:***@{host_path}"
    except Exception:
        # Fallback if URI parsing fails
        return "***"

    return db_uri


def run_agent_server():
    """Start the Kaval.AI agent server using environment configuration.

    The following environment variables are used:
    - KAVALAI_AGENT_WORKFLOW_PATH: Path to the workflow YAML file.
    - KAVALAI_DB_URI: Database connection string.
    - KAVALAI_DB_SCHEMA: Database schema name.
    - KAVALAI_SQL_ECHO: Whether to log SQL queries (optional, default: False).
    - KAVALAI_AGENT_HOST: Host to bind the server to (optional, default: 0.0.0.0).
    - KAVALAI_AGENT_PORT: Port to bind the server to (optional, default: 10000).
    """
    workflow_path = env.str("KAVALAI_AGENT_WORKFLOW_PATH")
    logger.info(f"Loading workflow from {workflow_path}.")
    workflow = Workflow.from_yaml_path(workflow_path)

    # Log database connection info
    db_uri = env("KAVALAI_DB_URI")
    db_schema = env("KAVALAI_DB_SCHEMA", "public")

    masked_uri = mask_db_uri(db_uri)

    logger.info(f"Database URI: {masked_uri}")
    logger.info(f"Database Schema: {db_schema}")

    # Log basic auth info
    auth_user = env.str("KAVALAI_AGENT_BASIC_AUTH_USER", "")
    auth_password = env.str("KAVALAI_AGENT_BASIC_AUTH_USER_PASSWORD", "")

    if auth_user or auth_password:
        logger.info(f"Basic Auth configured for user: {auth_user}")
        if auth_password:
            logger.info("Basic Auth password: ***")
    else:
        logger.warning("Basic Auth is NOT configured. Server is public.")

    # Specify the connection to the KavalAI agent server.
    session_provider = db_manager.get_sessionmaker(
        uri=db_uri, echo=env.bool("KAVALAI_SQL_ECHO", False)
    )

    # Create FastAPI app.
    app = create_agent_app(
        workflow=workflow,
        session_provider=session_provider,
    )

    logger.info(f"Starting agent <{workflow.workflow_model.name}>.")
    uvicorn.run(
        app,
        host=env.str("KAVALAI_AGENT_HOST", "0.0.0.0"),
        port=env.int("KAVALAI_AGENT_PORT", 10000),
    )


if __name__ == "__main__":
    run_agent_server()
