"""Launch Kaval.AI agent REST server.

Example usage:
HTTP_BASIC_AUTH_USER=user HTTP_BASIC_AUTH_PASSWORD=password python -m kavalai.agents.server kavalai/demo_agents/silverhand.yaml --port 10000
"""

import logging
import os
import secrets
from argparse import ArgumentParser
from contextlib import asynccontextmanager
from typing import Annotated
from typing import Optional, Union
from uuid import UUID

import uvicorn
from fastapi import Depends
from fastapi import HTTPException, status, FastAPI, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import async_sessionmaker

from kavalai.agents.agent_service import AgentService
from kavalai.agents.db import db_manager
from kavalai.agents.workflow import Workflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

security = HTTPBasic()


def validate_auth(credentials: Annotated[HTTPBasicCredentials, Depends(security)]):
    expected_username = os.environ.get("HTTP_BASIC_AUTH_USER")
    expected_password = os.environ.get("HTTP_BASIC_AUTH_PASSWORD")

    if expected_username and expected_password:
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
    if isinstance(session_or_factory, async_sessionmaker):
        async with session_or_factory() as session:
            yield session
    else:
        yield session_or_factory


def create_agent_app(
    workflow: Workflow,
    session_provider: Union[async_sessionmaker, None] = None,
) -> FastAPI:
    app = FastAPI(
        title=workflow.workflow_model.name,
        description=workflow.workflow_model.description,
    )

    InputDataType = workflow.get_data_type("input")
    OutputDataType = workflow.get_data_type("output")

    class InputType(BaseModel):
        session_id: Optional[UUID] = None
        external_id: Optional[str] = None
        data: InputDataType

    class OutputType(BaseModel):
        session_id: Optional[UUID]
        data: OutputDataType

    @app.post("/run_agent", response_model=OutputType, operation_id="run_agent")
    async def run_agent(
        input_data: InputType,
        credentials: Annotated[HTTPBasicCredentials, Depends(security)],
    ) -> OutputType:
        validate_auth(credentials)

        async with session_scope(session_provider) as session:
            workflow.agent_service = AgentService(session)
            result = await workflow.run(
                input_data=input_data.data.model_dump(),
                session_id=input_data.session_id,
                external_id=input_data.external_id,
            )
            return OutputType(session_id=result.session_id, data=result.data)

    @app.get("/workflow", response_model=OutputType, operation_id="run_agent")
    async def get_workflow(
        credentials: Annotated[HTTPBasicCredentials, Depends(security)],
    ):
        validate_auth(credentials)
        return Response(
            content=workflow.workflow_model.model_dump_json(),
            media_type="application/json",
        )

    return app


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Kaval.AI Agent REST Server. Uses HTTP_BASIC_AUTH_USER and HTTP_BASIC_AUTH_PASSWORD environment variables for authentication."
    )
    parser.add_argument("workflow_yaml_path", type=str, help="Path to workflow YAML")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument(
        "--sql-echo", action="store_true", help="Enable SQLAlchemy engine logging"
    )
    args = parser.parse_args()

    workflow = Workflow.from_yaml_path(args.workflow_yaml_path)
    uri = os.environ.get("KAVALAI_DB_URI")
    if uri:
        session_provider = db_manager.get_sessionmaker(uri=uri, echo=args.sql_echo)
    else:
        session_provider = db_manager.get_sessionmaker(
            user=os.environ["AGENTS_DB_USER"],
            password=os.environ["AGENTS_DB_PASSWORD"],
            host=os.environ["AGENTS_DB_HOST"],
            port=int(os.environ["AGENTS_DB_PORT"]),
            db_name=os.environ["AGENTS_DB_NAME"],
            echo=args.sql_echo,
        )

    app = create_agent_app(
        workflow=workflow,
        session_provider=session_provider,
    )
    logger.info(f"Starting <{workflow.workflow_model.name}>.")
    uvicorn.run(app, host=args.host, port=args.port)
