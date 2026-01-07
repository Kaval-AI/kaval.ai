import logging
import os
import secrets
from argparse import ArgumentParser
from contextlib import asynccontextmanager
from typing import Optional
from typing import Union
from uuid import UUID

import uvicorn
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastmcp import FastMCP
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import async_sessionmaker

from kavalai.agents.agent_service import AgentService
from kavalai.agents.db import AsyncAgentsSession
from kavalai.agents.workflow import Workflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def authenticate(credentials: HTTPBasicCredentials = Depends(HTTPBasic())):
    expected_username = os.environ.get("BASIC_AUTH_USERNAME")
    expected_password = os.environ.get("BASIC_AUTH_PASSWORD")

    if not expected_username or not expected_password:
        logger.warning("Auth credentials not set in environment variables")
        raise HTTPException(status_code=500, detail="Server auth configuration missing")

    is_correct_username = secrets.compare_digest(
        credentials.username, expected_username
    )
    is_correct_password = secrets.compare_digest(
        credentials.password, expected_password
    )

    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


@asynccontextmanager
async def session_scope(session_or_factory):
    if isinstance(session_or_factory, async_sessionmaker):
        async with session_or_factory() as session:
            yield session
    else:
        yield session_or_factory


def create_mcp_agent_server(
    workflow: Workflow,
    session_provider: Union[AsyncAgentsSession, async_sessionmaker, None] = None,
) -> FastMCP:
    mcp = FastMCP(workflow.workflow_model.name)
    InputDataType = workflow.get_data_type("input")
    OutputDataType = workflow.get_data_type("output")

    class InputType(BaseModel):
        session_id: Optional[UUID] = None
        external_id: Optional[str] = None
        data: InputDataType

    class OutputType(BaseModel):
        session_id: Optional[UUID]
        data: OutputDataType

    @mcp.tool()
    async def run_agent(input_data: InputType) -> OutputType:
        async with session_scope(session_provider) as session:
            workflow.agent_service = AgentService(session)
            result = await workflow.run(
                input_data=input_data.data.model_dump(),
                session_id=input_data.session_id,
            )
            return OutputType(session_id=result.session_id, data=result.data)

    return mcp


if __name__ == "__main__":
    parser = ArgumentParser(description="Kaval.AI Agent MCP Server")
    parser.add_argument("workflow_yaml_path", type=str, help="Path to workflow YAML")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    app = create_mcp_agent_server(Workflow.from_yaml_path(args.workflow_yaml_path))
    uvicorn.run(app, host=args.host, port=args.port)
