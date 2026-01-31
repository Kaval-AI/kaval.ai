"""
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
from typing import Dict, Type, Optional, Literal
from uuid import UUID

import httpx
import yaml
from environs import Env
from pydantic import BaseModel

from kavalai.agents.agent_service import AgentService
from kavalai.agents.schema_parser import SchemaParser
from kavalai.llm_clients.common import chat_completions

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
    use_history: bool = True
    # REST tool call
    tool: Optional[str] = None
    rest_server: Optional[str] = None


class RestServer(BaseModel):
    name: str
    url: str
    username: Optional[str] = None
    password: Optional[str] = None


class WorkflowModel(BaseModel):
    name: str
    description: str = ""
    version: str = "1.0"
    llm_model: Optional[str] = None
    data_types: dict[str, dict]
    rest_servers: list[RestServer] = []
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
    run_context: Optional[RunContext] = None


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
        self.rest_servers = {
            server.name: server for server in workflow_model.rest_servers
        }
        self.tasks = {task.name: task for task in workflow_model.tasks}
        self.validate_workflow()
        self.env = Env()
        self.env.read_env()

    def validate_workflow(self):
        available_data = {"input"}
        for task in self.workflow_model.tasks:
            # Check outputs
            if task.output not in self.workflow_model.data_types:
                raise WorkflowException(
                    f"output '{task.output}' in task '{task.name}' is not defined in data_types."
                )

            # Check inputs
            for input_name, input_info in task.inputs.items():
                if input_info.type == "context":
                    # If 'name' is provided, that's what we look for in available_data
                    actual_input_name = input_info.name or input_name
                    if actual_input_name not in available_data:
                        raise WorkflowException(
                            f"input '{actual_input_name}' in task '{task.name}' is not available. "
                            f"Available context: {sorted(list(available_data))}"
                        )
            # After task success, its output is available for next tasks
            available_data.add(task.output)

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
        input_data = {}
        for name, info in task.inputs.items():
            if info.type == "literal":
                input_data[name] = info.value
            else:
                input_data[name] = run_context.data.get(info.name or name)

        input_text = make_prompt(task.prompt, input_data)

        session = self.agent_service.db if self.agent_service else None

        system_message = dict(role="system", content=input_text)
        messages = [system_message]

        if task.use_history and self.agent_service and run_context.session_id:
            history = await self.agent_service.get_chat_history(run_context.session_id)
            for msg in history:
                messages.append(dict(role=msg.role, content=msg.content))

        llm_model = self.workflow_model.llm_model or self.env.str("DEFAULT_LLM_MODEL")
        response, stats = await chat_completions(
            model=llm_model,
            response_model=self.get_data_type(task.output),
            messages=messages,
        )
        if session:
            session.add(stats)

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

        rest_server = self.rest_servers[task.rest_server]
        auth = (
            (rest_server.username, rest_server.password)
            if rest_server.username and rest_server.password
            else None
        )

        async with httpx.AsyncClient(auth=auth) as client:
            response = await client.get(
                f"{rest_server.url}/{task.tool}", params=inputs, timeout=60.0
            )
            response.raise_for_status()
            result_data = response.json()

        output_type = self.get_data_type(task.output)
        if output_type:
            result = (
                output_type(**result_data)
                if isinstance(result_data, dict)
                else result_data
            )
        else:
            result = result_data
        debug_data = str(result)[:50]
        logger.info(f"Setting {task.output} = {debug_data}")
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
                input_schema=self.workflow_model.data_types.get("input"),
                output_schema=self.workflow_model.data_types.get("output"),
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

        return WorkflowRunResult(
            session_id=run_context.session_id,
            data=output_model,
            run_context=run_context,
        )
