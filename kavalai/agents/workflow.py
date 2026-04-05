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

from loguru import logger
from typing import Dict, Type, Optional
from uuid import UUID

import yaml
import json
from environs import Env
from pydantic import BaseModel, ValidationError
import time

from kavalai.agents.workflow_model import (
    WorkflowModel,
    LLMTask,
    RestTask,
    McpTask,
    PythonTask,
    AgentTask,
    AssignTask,
    RagQueryTask,
    WorkflowRunResult,
    WorkflowException,
    Task,
)
from kavalai.agents.utils import to_plain
from kavalai.agents.planning_agent import PlanningAgent
from kavalai.agents.agent_service import AgentService
from kavalai.agents.schema_parser import SchemaParser
from kavalai.agents.workflow_validation import (
    validate_rest_server_env_vars,
    validate_workflow,
)
from kavalai.agents.rag_service import RagService
from kavalai.agents.task_logger import TaskLogger
from kavalai.llm_clients.llm_client import LLMClient
from kavalai.llm_clients.common import Streamer
from kavalai.agents.run_context import RunContext
from kavalai.functionkernel import FunctionKernel, pythontool
import asyncio
import importlib


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
    """Workflow represents the central element in the agent orchestration.
    It manages running the tasks, storing the results and streaming the results of the agent run.

    Parameters
    ==========
    workflow_model: WorkflowModel
        The workflow model that defines the structure and behavior of the workflow.
    agent_service: Optional[AgentService]
        The agent service for storing results in a DB.
    yaml_content: Optional[str]
        The YAML content associated with the workflow (if given, we can give more accurate exceptions).
    """

    def __init__(
        self,
        workflow_model: WorkflowModel,
        agent_service: Optional[AgentService] = None,
        yaml_content: Optional[str] = None,
    ):
        self.workflow_model = workflow_model
        self.agent_service = agent_service
        self.yaml_content = yaml_content
        self.parser = SchemaParser(workflow_model.data_types)
        self.models: Dict[str, Type[BaseModel]] = self.parser.parse_all()
        self.tasks = {task.name: task for task in workflow_model.tasks}
        validate_workflow(self.workflow_model)
        self.env = Env()
        self.env.read_env()
        validate_rest_server_env_vars(self.workflow_model)
        self.run_context = RunContext(agent_service=self.agent_service)
        self.task_logger = TaskLogger(
            agent_service=self.agent_service, run_context=self.run_context
        )

        # Initialize FunctionKernel and register all servers
        self.kernel = FunctionKernel()
        for server in workflow_model.rest_servers:
            self.kernel.register_rest_server(server)
        for server in workflow_model.mcp_servers:
            self.kernel.register_mcp_server(server)

        # Register Python functions
        for func_config in workflow_model.python_functions:
            module_path, func_name = func_config.path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
            # Ensure it's marked as a kavalai tool
            if not getattr(func, "_is_kavalai_tool", False):
                func = pythontool(func)

            self.kernel.register_python_tool(func_config.name, func)

        # Register REST tools explicitly if specified in tasks
        for task in workflow_model.tasks:
            if isinstance(task, RestTask):
                input_model = self.models.get(f"{task.name}_input")
                output_model = self.models.get(task.output)
                if input_model and output_model:
                    self.kernel.register_rest_tool(
                        server_name=task.rest_server,
                        tool_name=task.tool,
                        method=task.method,
                        input_schema=input_model.model_json_schema(),
                        output_schema=output_model.model_json_schema(),
                    )

    @classmethod
    def from_yaml_path(
        cls, yaml_path: str, agent_service: Optional[AgentService] = None
    ):
        """Initiates a workflow instance from workflow model YAML file.

        Parameters
        ==========
        yaml_path: str
            The path to the YAML file containing the workflow model.
        agent_service: Optional[AgentService]
            The agent service for storing results in a DB.
        """
        with open(yaml_path, "r") as f:
            yaml_string = f.read()
            return cls.from_yaml(yaml_string, agent_service=agent_service)

    @classmethod
    def from_yaml(cls, yaml_string: str, agent_service: Optional[AgentService] = None):
        """Initiates a workflow instance from workflow model YAML string.

        Parameters
        ==========
        yaml_string: str
            The YAML string containing the workflow model.
        agent_service: Optional[AgentService]
            The agent service for storing results in a DB.
        """
        try:
            data = yaml.load(yaml_string, Loader=yaml.SafeLoader)  # nosec B506
            workflow_model = WorkflowModel(**data)
            return cls(
                workflow_model, agent_service=agent_service, yaml_content=yaml_string
            )
        except ValidationError as e:
            raise WorkflowException(f"Workflow validation failed: {e}") from e

    def get_data_type(self, name: str) -> Type[BaseModel]:
        """Retrieve a a real instance of a Pydantic model that was defined in the YAML file."""
        if name not in self.models:
            raise KeyError(f"Data type '{name}' was not defined in YAML datatypes.")
        return self.models[name]

    async def run_llm_task(
        self,
        task: LLMTask,
        run_context: RunContext,
        queue: asyncio.Queue | None,
    ):
        input_data = await run_context.prepare_tool_inputs(task)

        # Render prompt with templates and context and construct message history
        rendered_prompt = await run_context.render_prompt(task.prompt)
        input_text = make_prompt(rendered_prompt, input_data)
        system_message = dict(role="system", content=input_text)
        message_history = [system_message]

        # Retrieve message history and append them to messages.
        if task.use_history and self.agent_service and run_context.session_id:
            history = await self.agent_service.get_chat_history(run_context.session_id)
            for msg in history:
                message_history.append(dict(role=msg.role, content=msg.content))

        # Retrive the LLM model
        llm_model = (
            task.llm_model
            if task.llm_model
            else (
                self.workflow_model.llm_model
                or self.env.str("KAVALAI_DEFAULT_LLM_MODEL")
            )
        )

        # Prepare LLM model keyword arguments.
        llm_kwargs = self.workflow_model.llm_kwargs.copy()
        llm_kwargs.update(task.llm_kwargs)

        # Define streamer for sending back results from the LLM call.
        streamer = None
        if task.stream_output and queue is not None:
            streamer = Streamer(task.output, queue)

        client = LLMClient(model=llm_model)
        start_time = time.perf_counter()
        response, stats = await client.chat_completions(
            response_model=self.get_data_type(task.output),
            messages=message_history,
            streamer=streamer,
            **llm_kwargs,
        )
        llm_call_duration = time.perf_counter() - start_time

        # Save the LLM call stats if agent_service is defined.
        if self.agent_service:
            await self.agent_service.add_model_call_stats(
                stats=stats, agent_id=run_context.agent_id
            )

        # We store the result in specified output variable.
        logger.info(f"Setting {task.output} = {response}")
        run_context.data[task.output] = response

        # Log the task info in DB (fire-and-forget).
        self.task_logger.log_llm_task(
            task_name=task.name,
            prompt=input_text,
            input_data=input_data,
            output=response.model_dump()
            if isinstance(response, BaseModel)
            else {"result": response},
            duration=llm_call_duration,
        )

    async def run_rest_tool(self, task: RestTask, run_context: RunContext):
        inputs = await run_context.prepare_tool_inputs(task)

        # Use FunctionKernel to call REST tool
        tool_uri = f"rest://{task.rest_server}.{task.tool}"
        output_type = self.get_data_type(task.output)

        start_time = time.perf_counter()
        result = await self.kernel.call_tool(
            tool_uri=tool_uri,
            arguments=inputs,
            output_type=output_type,
            method=task.method,
        )
        duration = time.perf_counter() - start_time

        # Store the output data
        debug_data = str(result)[:50]
        logger.info(f"Setting {task.output} = {debug_data}")
        run_context.data[task.output] = result

        # Store the tool run info (fire-and-forget).
        tool_uri = f"rest://{task.rest_server}.{task.tool}"
        self.task_logger.log_tool_call(
            tool_uri=tool_uri,
            arguments=inputs,
            output=result.model_dump() if isinstance(result, BaseModel) else result,
            duration=duration,
        )

    async def run_mcp_tool(self, task: McpTask, run_context: RunContext):
        inputs = await run_context.prepare_tool_inputs(task)

        # Use FunctionKernel to call MCP tool
        tool_uri = f"mcp://{task.mcp_server}.{task.tool}"
        output_type = self.get_data_type(task.output)

        start_time = time.perf_counter()
        result = await self.kernel.call_tool(
            tool_uri=tool_uri,
            arguments=inputs,
            output_type=output_type,
        )
        duration = time.perf_counter() - start_time

        debug_data = str(result)[:50]
        logger.info(f"Setting {task.output} = {debug_data}")
        run_context.data[task.output] = result

        # Store the tool run info (fire-and-forget).
        tool_uri = f"mcp://{task.mcp_server}.{task.tool}"
        self.task_logger.log_tool_call(
            tool_uri=tool_uri,
            arguments=inputs,
            output=result.model_dump() if isinstance(result, BaseModel) else result,
            duration=duration,
        )

    async def run_python_tool(
        self, task: PythonTask, run_context: RunContext, queue: asyncio.Queue | None
    ):
        """Run a Python function using FunctionKernel."""
        if not task.python_tool:
            raise WorkflowException(f"Task '{task.name}' has no python_tool defined.")

        # Resolve inputs
        inputs = await run_context.prepare_tool_inputs(task)

        # Use FunctionKernel to call Python tool
        tool_uri = f"python://{task.python_tool}"
        output_type = self.get_data_type(task.output) if task.output else None

        start_time = time.perf_counter()
        result = await self.kernel.call_tool(
            tool_uri=tool_uri,
            arguments=inputs,
            output_type=output_type,
        )
        duration = time.perf_counter() - start_time

        if task.output:
            run_context.data[task.output] = result
            if task.stream_output and queue is not None:
                streamer = Streamer(task.output, queue)
                await streamer.stream_complete(
                    result.model_dump_json()
                    if hasattr(result, "model_dump_json")
                    else str(result)
                )

        # Record in DB (fire-and-forget)
        tool_uri = f"python://{task.python_tool}"
        self.task_logger.log_tool_call(
            tool_uri=tool_uri,
            arguments=inputs,
            output=to_plain(result),
            duration=duration,
        )

    async def run_assign(self, task: AssignTask, run_context: RunContext):
        """Assigns a variable by combining various inputs."""
        # If output is a string, it means we are combining inputs into a named data type
        result = {}
        for input_name, info in task.inputs.items():
            if info.value is None and info.name is None:
                info = info.model_copy(update={"value": input_name})
            val = await run_context.resolve_input_info(info)
            result[input_name] = to_plain(val)
        output_type = self.get_data_type(task.output)
        run_context.data[task.output] = output_type(**result)
        logger.info(f"Assign {task.output}")

    async def run_agent_task(
        self, task: AgentTask, run_context: RunContext, queue: asyncio.Queue | None
    ):
        """Invoke the PlanningAgent for complex multi-step tasks."""
        # Prepares input data.
        input_data = await run_context.prepare_tool_inputs(task)
        response_model = self.get_data_type(task.output)

        # LLM-Task specific output streamer.
        streamer = None
        if (
            task.stream_output or task.stream_persisted or task.stream_updates
        ) and queue is not None:
            streamer = Streamer(task.output, queue)

        llm_model = (
            task.llm_model
            if task.llm_model
            else (
                self.workflow_model.llm_model
                or self.env.str("KAVALAI_DEFAULT_LLM_MODEL")
            )
        )
        llm_client = LLMClient(model=llm_model)

        # Update llm kwargs
        llm_kwargs = self.workflow_model.llm_kwargs.copy()
        llm_kwargs.update(task.llm_kwargs)

        # Define task
        planning_agent = PlanningAgent(
            kernel=self.kernel,
            run_context=run_context,
            llm_client=llm_client,
            input_data=input_data,
            response_model=response_model,
            agent_service=self.agent_service,
            task_logger=self.task_logger,
            streamer=streamer,
            stream_updates=task.stream_updates,
            stream_output=task.stream_output,
            stream_persisted=task.stream_persisted,
            allowed_tools=task.allowed_tools,
            llm_kwargs=llm_kwargs,
        )

        # Set up chat message history.
        chat_history = []
        if task.use_history and self.agent_service and run_context.session_id:
            history = await self.agent_service.get_chat_history(run_context.session_id)
            for msg in history:
                chat_history.append({"role": msg.role, "content": msg.content})

        # Render the prompt.
        rendered_prompt = await run_context.render_prompt(task.prompt)
        result = await planning_agent.run(
            task_name=task.name,
            task=rendered_prompt,
            chat_history=chat_history,
            max_iterations=task.max_steps,
        )

        run_context.data[task.name] = result
        run_context.data[task.output] = result

    async def run_rag_task(self, task: RagQueryTask, run_context: RunContext):
        """Perform a RAG search and store the results in the run context."""
        if not self.agent_service:
            logger.warning("AgentService not provided, skipping RAG task.")
            return

        # Resolve input text (it can be a literal or a context reference)
        text = task.text
        if "." in text:
            resolved = run_context.resolve_context_value(text)
            if resolved is not None:
                text = str(resolved)
        elif text in run_context.data:
            resolved = run_context.data[text]
            if resolved is not None:
                text = str(resolved)

        duration = 0.0
        if not text:
            run_context.data[task.name] = []
        else:
            model = self.workflow_model.embedding_model

            rag_service = RagService.from_session_maker(
                self.agent_service.session_maker, model
            )
            start_time = time.perf_counter()
            results = await rag_service.query(
                text=text,
                top_k=task.top_k,
                collection_name=task.collection_name,
                source_ids=task.source_ids,
                keep_best=task.keep_best,
            )
            duration = time.perf_counter() - start_time

            # 3. Store results in run_context.data (similarity, content, source_id, and metadata)
            run_context.data[task.name] = [
                {
                    "similarity": r.similarity,
                    "content": r.content,
                    "source_id": r.source_id,
                    "metadata": json.dumps(r.rag_metadata) if r.rag_metadata else None,
                }
                for r in results
            ]

        # Handle output mapping if defined
        run_context.data[task.output] = run_context.data[task.name]

        # Store the tool run info (fire-and-forget).
        self.task_logger.log_rag_query(
            task_name=task.name,
            query_text=text,
            top_k=task.top_k,
            collection_name=task.collection_name,
            source_ids=task.source_ids,
            keep_best=task.keep_best,
            output=run_context.data[task.name],
            duration=duration,
        )

    async def _initialize_agent_session(
        self,
        *,
        session_id: Optional[UUID] = None,
        external_id: Optional[str] = None,
        input_data: dict,
    ):
        # Get or create the agent definition
        agent = await self.agent_service.get_or_create_agent(
            name=self.workflow_model.name,
            description=self.workflow_model.description,
            input_schema=self.workflow_model.data_types.get("input"),
            output_schema=self.workflow_model.data_types.get("output"),
            workflow=self.workflow_model.model_dump(),
        )
        self.run_context.agent_id = agent.id

        # Get or create session (using UUID or string external_id).
        # Not to be confused with sqlalchemy sessions.
        if session_id:
            session = await self.agent_service.get_or_create_session(
                agent_id=agent.id, session_id=session_id
            )
            if not session:
                raise WorkflowException(f"Session with ID {session_id} not found")
            self.run_context.session_id = session.id
        else:
            session = await self.agent_service.get_or_create_session(
                agent_id=agent.id, session_id=None, external_id=external_id
            )
            self.run_context.session_id = session.id

        # Creates the specific Run record for this execution
        run = await self.agent_service.create_run(
            session_id=self.run_context.session_id, input_data=input_data
        )
        self.run_context.run_id = run.id

    async def _execute_task(
        self,
        streamer: Streamer | None,
        task: Task,
        run_context: RunContext,
        queue: asyncio.Queue,
    ) -> bool:
        """Executes a single task in the workflow.

        Returns True if the task should be stopped.
        """
        # Stream the current runnig task.
        if streamer and task.stream_updates:
            await streamer.stream_complete(task.name, name="running_task")

        # Decide if we should skip this task
        if task.when:
            if not await run_context.evaluate_condition(task.when):
                logger.info(f"Skipping task <{task.name}> due to condition")
                return

        logger.info(f"Running task {task.name}")
        try:
            if isinstance(task, AgentTask):
                await self.run_agent_task(task, run_context, queue)
            elif isinstance(task, LLMTask):
                await self.run_llm_task(task, run_context, queue)
            elif isinstance(task, McpTask):
                await self.run_mcp_tool(task, run_context)
            elif isinstance(task, PythonTask):
                await self.run_python_tool(task, run_context, queue)
            elif isinstance(task, RestTask):
                await self.run_rest_tool(task, run_context)
            elif isinstance(task, AssignTask):
                await self.run_assign(task, run_context)
            elif isinstance(task, RagQueryTask):
                await self.run_rag_task(task, run_context)
            else:
                logger.warning(f"Unknown task type: {type(task)}")
        except Exception as e:
            raise WorkflowException(e) from e
        return task.stop

    async def run(
        self,
        input_data: dict,
        session_id: Optional[UUID] = None,
        external_id: Optional[str] = None,
        queue: asyncio.Queue | None = None,
    ) -> WorkflowRunResult:
        agent_service = self.agent_service
        parsed_input = self.get_data_type("input")(**input_data)
        run_context = self.run_context
        run_context.data["input"] = parsed_input
        run_context.templates = {t.name: t.value for t in self.workflow_model.templates}

        # Initialize DB data if agent_service is given.
        if agent_service:
            await self._initialize_agent_session(
                session_id=session_id,
                external_id=external_id,
                input_data=input_data,
            )
            # Store the user message in agent_service.
            user_msg = getattr(parsed_input, "user_message", str(input_data))
            await agent_service.add_chat_message(
                agent_id=run_context.agent_id,
                session_id=run_context.session_id,
                run_id=run_context.run_id,
                role="user",
                content=user_msg,
            )
        # Executes the workflow steps
        try:
            for task in self.workflow_model.tasks:
                streamer = Streamer(name=task.output, queue=queue) if queue else None
                # Execute the step and stop the execution if the signals so.
                stop_execution = await self._execute_task(
                    streamer=streamer, task=task, run_context=run_context, queue=queue
                )
                if stop_execution:
                    logger.info(
                        f"Stopping workflow after task <{task.name}> due to stop: True"
                    )
                    break
        finally:
            # Cleanup MCP sessions/clients using FunctionKernel
            await self.kernel.close()

        # Update database
        output_data = run_context.data.get("output")

        if agent_service:
            # Persist final output_data and full execution context into the Run
            serialized_context = to_plain(run_context.data)
            serialized_output = (
                to_plain(output_data) if output_data is not None else None
            )

            await agent_service.update_run(
                run_id=run_context.run_id,
                output_data=serialized_output,
                context=serialized_context,
            )

            # Also log assistant message if we have a final output with a textual response
            if output_data is not None:
                agent_resp = getattr(output_data, "agent_response", "")
                await agent_service.add_chat_message(
                    agent_id=run_context.agent_id,
                    session_id=run_context.session_id,
                    run_id=run_context.run_id,
                    role="assistant",
                    content=agent_resp,
                )

        logger.info(
            f"Workflow completed successfully for session_id={run_context.session_id}"
        )

        return WorkflowRunResult(
            session_id=run_context.session_id,
            data=output_data,
            run_context=run_context,
        )
