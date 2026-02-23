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
import os
from typing import Dict, Type, Optional
from uuid import UUID

import httpx
import yaml
from environs import Env
from pydantic import BaseModel

from kavalai.agents.workflow_model import (
    WorkflowModel,
    Task,
    to_plain,
    WorkflowRunResult,
)
from kavalai.agents.agent_service import AgentService
from kavalai.agents.schema_parser import SchemaParser
from kavalai.agents.workflow_validation import (
    validate_rest_server_env_vars,
    validate_workflow,
)
from kavalai.llm_clients.llm_client import chat_completions
from kavalai.llm_clients.common import Streamer
from kavalai.agents.run_context import RunContext
import asyncio

logger = logging.getLogger(__name__)


def make_prompt(prompt: str, input_data: dict) -> str:
    pieces = [prompt]
    if len(input_data) > 0:
        pieces.append("INPUT DATA:")
        for key, value in input_data.items():
            if isinstance(value, BaseModel):
                value = value.model_dump_json()
            pieces.append(f"{key}:{value}")
    return "\n".join(pieces)


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
        validate_workflow(self.workflow_model)
        self.env = Env()
        self.env.read_env()
        validate_rest_server_env_vars(self.workflow_model)

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

    async def run_prompt(
        self,
        task: Task,
        run_context: RunContext,
        queue: asyncio.Queue | None,
        agent_service: Optional[AgentService] = None,
    ):
        agent_service = agent_service or self.agent_service
        input_data = {}
        for name, info in task.inputs.items():
            if info.value is None and info.name is None:
                info = info.model_copy(update={"value": name})
            input_data[name] = await run_context.resolve_input_info(info)

        input_text = make_prompt(task.prompt, input_data)

        session = agent_service.db if agent_service else None

        system_message = dict(role="system", content=input_text)
        messages = [system_message]

        if task.use_history and agent_service and run_context.session_id:
            history = await agent_service.get_chat_history(run_context.session_id)
            for msg in history:
                messages.append(dict(role=msg.role, content=msg.content))

        llm_model = self.workflow_model.llm_model or self.env.str(
            "KAVALAI_DEFAULT_LLM_MODEL"
        )

        temperature = (
            task.temperature
            if task.temperature is not None
            else self.workflow_model.temperature
        )

        streamer = None
        if task.stream and queue is not None:
            streamer = Streamer(task.output, queue)

        response, stats = await chat_completions(
            model=llm_model,
            response_model=self.get_data_type(task.output),
            messages=messages,
            streamer=streamer,
            temperature=temperature,
        )
        if session:
            session.add(stats)

        logger.info(f"Setting {task.output} = {response}")
        run_context.data[task.output] = response

        # DB LOGGING: Record this prompt as a Task
        if agent_service and run_context.run_id:
            await agent_service.add_task(
                agent_id=run_context.agent_id,
                session_id=run_context.session_id,
                run_id=run_context.run_id,
                inputs={"prompt": input_text},
                output=response.model_dump()
                if isinstance(response, BaseModel)
                else {"result": response},
            )

    async def run_tool(
        self,
        task: Task,
        run_context: RunContext,
        queue: asyncio.Queue | None,
        agent_service: Optional[AgentService] = None,
    ):
        agent_service = agent_service or self.agent_service
        inputs = await run_context.prepare_tool_inputs(task)

        rest_server = self.rest_servers[task.rest_server]
        url = rest_server.url
        if not url and rest_server.url_env:
            url = os.environ[rest_server.url_env]

        auth = None
        if rest_server.username_env and rest_server.password_env:
            username = os.environ[rest_server.username_env]
            password = os.environ[rest_server.password_env]
            auth = (username, password)

        async with httpx.AsyncClient(auth=auth) as client:
            kwargs = {
                "params": inputs,
                "timeout": 60.0,
            }
            if task.method.lower() in ("post", "put", "patch"):
                kwargs["json"] = inputs
                if "params" in kwargs:
                    kwargs.pop("params")
            logger.info(f"Calling {task.method.upper()} {url}/{task.tool}")
            response = await client.request(
                task.method.upper(),
                f"{url}/{task.tool}",
                **kwargs,
            )
            response.raise_for_status()
            result_data = response.json()

        # Convert result to output type
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

        # Publish to stream
        if task.stream and queue is not None:
            streamer = Streamer(task.output, queue)
            stream_value = (
                result.model_dump_json()
                if isinstance(result, BaseModel)
                else str(result)
            )
            await streamer.stream_complete(stream_value)

        # Store the tool run info.
        if agent_service and run_context.run_id:
            await agent_service.add_task(
                agent_id=run_context.agent_id,
                session_id=run_context.session_id,
                run_id=run_context.run_id,
                inputs={"tool": task.tool, "arguments": inputs},
                output=result.model_dump() if isinstance(result, BaseModel) else result,
            )

    async def run_combine(
        self, task: Task, run_context: RunContext, queue: asyncio.Queue | None = None
    ):
        """Combine context values into an output dict (no LLM or tool call)."""
        result = {}
        if isinstance(task.output, dict):
            for field_name, info in task.output.items():
                if info.value is None and info.name is None:
                    info = info.model_copy(update={"value": field_name})
                result[field_name] = await run_context.resolve_input_info(info)
            # Store as 'output' in context (standard output key for final result)
            output_type = self.get_data_type("output")
            model_instance = output_type(**result)
            run_context.data["output"] = model_instance
            logger.info(f"Combined output with fields: {list(result.keys())}")
            if task.stream and queue is not None:
                streamer = Streamer("output", queue)
                await streamer.stream_complete(model_instance.model_dump_json())
        elif isinstance(task.output, str):
            # If output is a string, it means we are combining inputs into a named data type
            for input_name, info in task.inputs.items():
                if info.value is None and info.name is None:
                    info = info.model_copy(update={"value": input_name})
                result[input_name] = await run_context.resolve_input_info(info)
            output_type = self.get_data_type(task.output)
            run_context.data[task.output] = output_type(**result)
            if task.stream and queue is not None:
                streamer = Streamer(task.output, queue)
                await streamer.stream_complete(
                    run_context.data[task.output].model_dump_json()
                )
            logger.info(f"Combined inputs into {task.output}")

    async def run(
        self,
        input_data: dict,
        session_id: Optional[UUID] = None,
        external_id: Optional[str] = None,
        queue: asyncio.Queue | None = None,
        agent_service: Optional[AgentService] = None,
    ) -> WorkflowRunResult:
        agent_service = agent_service or self.agent_service
        # 1. Parse Input
        parsed_input = self.get_data_type("input")(**input_data)
        run_context = RunContext(agent_service=agent_service)
        run_context.data["input"] = parsed_input

        # 2. Initialize DB Context (Agent -> Session -> Run)
        if agent_service:
            # Get or create the agent definition
            agent = await agent_service.get_or_create_agent(
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
                session = await agent_service.get_or_create_session(
                    agent_id=agent.id, session_id=session_id, external_id=external_id
                )
                run_context.session_id = session.id

            # Create the specific Run record for this execution
            run = await agent_service.create_run(
                session_id=run_context.session_id, input_data=input_data
            )
            run_context.run_id = run.id

            # Log the initial user message
            user_msg = getattr(parsed_input, "user_message", str(input_data))
            await agent_service.add_chat_message(
                agent_id=run_context.agent_id,
                session_id=run_context.session_id,
                run_id=run_context.run_id,
                role="user",
                content=user_msg,
            )

        # 3. Execute Workflow Steps
        for task in self.workflow_model.tasks:
            if task.when:
                if not await run_context.evaluate_condition(task.when):
                    logger.info("Skipping task <%s> due to condition", task.name)
                    continue

            logger.info("Running task <%s>", task.name)
            if task.prompt:
                await self.run_prompt(task, run_context, queue, agent_service)
            elif task.tool:
                await self.run_tool(task, run_context, queue, agent_service)
            else:
                await self.run_combine(task, run_context, queue)

            if task.stop:
                logger.info(
                    "Stopping workflow after task <%s> due to stop: True", task.name
                )
                break

        # 4. Finalize and Log Response
        output_model = run_context.data.get("output")
        if agent_service:
            # Persist final output_data and full execution context into the Run
            serialized_context = to_plain(run_context.data)
            serialized_output = (
                to_plain(output_model) if output_model is not None else None
            )

            await agent_service.update_run(
                run_id=run_context.run_id,
                output_data=serialized_output,
                context=serialized_context,
            )

            # Also log assistant message if we have a final output with a textual response
            if output_model is not None:
                agent_resp = getattr(output_model, "agent_response", str(output_model))
                await agent_service.add_chat_message(
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
