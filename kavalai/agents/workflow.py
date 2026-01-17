import logging
from typing import Dict, Type, Optional, Literal
from uuid import UUID

import yaml
from fastmcp import Client
from pydantic import BaseModel

from kavalai.agents.agent_service import AgentService
from kavalai.agents.db import upsert_llm_profile
from kavalai.agents.llm_config import (
    get_instructor,
    load_profile_from_path,
)
from kavalai.agents.schema_parser import SchemaParser

logger = logging.getLogger(__name__)


class TypeInputInfo(BaseModel):
    type: Literal["literal", "context"]
    value: Optional[BaseModel | str | int | float | bool] = None
    name: Optional[str] = None


class Task(BaseModel):
    name: str
    inputs: dict[str, TypeInputInfo] = {}
    output: str
    # LLM call
    prompt: Optional[str] = None
    # MCP tool call
    tool: Optional[str] = None
    mcp_server: Optional[str] = None


class McpServer(BaseModel):
    name: str
    url: str
    username: Optional[str] = None
    password: Optional[str] = None


class WorkflowModel(BaseModel):
    name: str
    description: str = ""
    llm_profile_name: str
    data_types: dict[str, dict]
    mcp_servers: list[McpServer] = []
    tasks: list[Task]


class RunContext(BaseModel):
    """Runtime data for a single interaction."""

    agent_id: Optional[UUID] = None
    session_id: Optional[UUID] = None
    run_id: Optional[UUID] = None
    data: dict = {}


def make_prompt(prompt: str, input_data: dict) -> str:
    pieces = [prompt]
    if len(input_data) > 0:
        pieces.append("INPUT DATA:")
        for key, value in input_data.items():
            if isinstance(value, BaseModel):
                value = value.model_dump_json()
            pieces.append(f"{key}:{value}")
    return "\n".join(pieces)


class WorkflowException(Exception):
    pass


class WorkflowRunResult(BaseModel):
    session_id: Optional[UUID] = None
    data: Optional[BaseModel] = None


class Workflow:
    def __init__(
        self,
        workflow_model: WorkflowModel,
        agent_service: Optional[AgentService] = None,
    ):
        self.workflow_model = workflow_model
        self.agent_service = agent_service
        self.parser = SchemaParser(workflow_model.data_types)
        self.models: Dict[str, Type[BaseModel]] = self.parser.parse_all()
        self.mcp_servers = {
            server.name: server for server in workflow_model.mcp_servers
        }
        self.tasks = {task.name: task for task in workflow_model.tasks}

    @classmethod
    def from_yaml_path(cls, yaml_path: str):
        with open(yaml_path, "r") as f:
            return Workflow.from_yaml(f.read())

    @classmethod
    def from_yaml(cls, yaml_string: str):
        workflow_model = WorkflowModel(**yaml.safe_load(yaml_string))
        return cls(workflow_model)

    def get_data_type(self, name: str) -> Type[BaseModel]:
        """Retrieve a generated Pydantic model by name."""
        if name not in self.models:
            raise KeyError(f"Data type '{name}' was not defined in YAML datatypes.")
        return self.models[name]

    async def run_prompt(self, task: Task, run_context: RunContext):
        input_data = {
            input_name: run_context.data.get(input_name) for input_name in task.inputs
        }
        input_text = make_prompt(task.prompt, input_data)

        session = self.agent_service.db if self.agent_service else None

        llm_profile = load_profile_from_path(self.workflow_model.llm_profile_name)
        if not llm_profile:
            raise Exception(
                f"LLM Profile '{self.workflow_model.llm_profile_name}' not found"
            )

        if session:
            await upsert_llm_profile(session, llm_profile)

        client = get_instructor(llm_profile)

        system_message = dict(role="system", content=input_text)
        response = await client.chat.completions.create(
            response_model=self.get_data_type(task.output),
            messages=[system_message],
        )

        run_context.data[task.output] = response

        # DB LOGGING: Record this prompt as a Task
        if self.agent_service and run_context.run_id:
            task_output = response
            if isinstance(response, dict):
                output_type = self.get_data_type(task.output)
                if output_type:
                    task_output = output_type(**response)
                    run_context.data[task.output] = task_output

            await self.agent_service.add_task(
                agent_id=run_context.agent_id,
                session_id=run_context.session_id,
                run_id=run_context.run_id,
                inputs={"prompt": input_text},
                output=task_output.model_dump()
                if isinstance(task_output, BaseModel)
                else {"result": task_output},
            )

    async def run_tool(self, task: Task, run_context: RunContext):
        inputs = {}
        for name, info in task.inputs.items():
            if info.type == "literal":
                inputs[name] = info.value
            else:
                # Fallback to the key name if info.name is not explicitly provided
                inputs[name] = run_context.data.get(info.name or name)

        server_url = self.mcp_servers[task.mcp_server].url
        async with Client(server_url) as client:
            result_data = await client.call_tool(task.tool, arguments=inputs)

        output_type = self.get_data_type(task.output)
        if output_type:
            result = (
                output_type(**result_data)
                if isinstance(result_data, dict)
                else result_data
            )
        else:
            result = result_data
        logger.info(f"Setting {task.output} = {result}")
        run_context.data[task.output] = result

        # Store the tool run info.
        if self.agent_service and run_context.run_id:
            await self.agent_service.add_task(
                agent_id=run_context.agent_id,
                session_id=run_context.session_id,
                run_id=run_context.run_id,
                inputs={"tool": task.tool, "arguments": inputs},
                output=result.model_dump() if isinstance(result, BaseModel) else result,
            )

    async def run(
        self,
        input_data: dict,
        session_id: Optional[UUID] = None,
        external_id: Optional[str] = None,
    ) -> WorkflowRunResult:
        # 1. Parse Input
        parsed_input = self.get_data_type("input")(**input_data)
        run_context = RunContext()
        run_context.data["input"] = parsed_input

        # 2. Initialize DB Context (Agent -> Session -> Run)
        if self.agent_service:
            # Get or create the agent definition
            agent = await self.agent_service.get_or_create_agent(
                name=self.workflow_model.name,
                description=self.workflow_model.description,
                workflow=self.workflow_model.model_dump(),
            )
            run_context.agent_id = agent.id

            # Get or create session (using UUID or string external_id)
            if session_id:
                run_context.session_id = session_id
            else:
                session = await self.agent_service.get_or_create_session(
                    agent_id=agent.id, session_id=session_id, external_id=external_id
                )
                run_context.session_id = session.id

            # Create the specific Run record for this execution
            run = await self.agent_service.create_run(
                session_id=run_context.session_id, input_data=input_data
            )
            run_context.run_id = run.id

            # Log the initial user message
            user_msg = getattr(parsed_input, "user_message", str(input_data))
            await self.agent_service.add_chat_message(
                agent_id=run_context.agent_id,
                session_id=run_context.session_id,
                run_id=run_context.run_id,
                role="user",
                content=user_msg,
            )

        # 3. Execute Workflow Steps
        for task in self.workflow_model.tasks:
            logger.info("Running task <%s>", task.name)
            if task.prompt:
                await self.run_prompt(task, run_context)
            elif task.tool:
                await self.run_tool(task, run_context)
            else:
                raise WorkflowException(
                    f"Task {task.name} has no prompt or tool defined."
                )

        # 4. Finalize and Log Response
        output_model = run_context.data.get("output")
        if self.agent_service and output_model:
            agent_resp = getattr(output_model, "agent_response", str(output_model))
            await self.agent_service.add_chat_message(
                agent_id=run_context.agent_id,
                session_id=run_context.session_id,
                run_id=run_context.run_id,
                role="assistant",
                content=agent_resp,
            )

        return WorkflowRunResult(session_id=run_context.session_id, data=output_model)
