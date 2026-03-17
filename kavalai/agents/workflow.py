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
from typing import Dict, Type, Optional, Any
from uuid import UUID

import yaml
from environs import Env
from pydantic import BaseModel, ValidationError

from kavalai.agents.workflow_model import (
    WorkflowModel,
    LLMTask,
    RestTask,
    McpTask,
    PythonTask,
    AgentTask,
    CombineTask,
    RagQueryTask,
    to_plain,
    WorkflowRunResult,
    WorkflowException,
)

from kavalai.agents.planning_agent import PlanningAgent
from kavalai.agents.agent_service import AgentService
from kavalai.agents.schema_parser import SchemaParser
from kavalai.agents.workflow_validation import (
    validate_rest_server_env_vars,
    validate_workflow,
)
from kavalai.agents.rag_service import RagService
from kavalai.llm_clients.llm_client import LLMClient
from kavalai.llm_clients.common import Streamer
from kavalai.agents.run_context import RunContext
from kavalai.functionkernel import FunctionKernel
import asyncio
import importlib


class LineLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = super().construct_mapping(node, deep=deep)
        mapping["__line__"] = node.start_mark.line + 1
        return mapping


def inject_metadata(data: Any, file_path: Optional[str] = None) -> Any:
    """Recursively inject __file_path__ into dictionaries that have __line__."""
    if isinstance(data, dict):
        # We must NOT inject metadata into dictionaries that are validated
        # by Pydantic as having a specific schema that doesn't include these fields.
        # This includes the root dictionary's keys that are not models.
        if "__line__" in data and file_path:
            data["__file_path__"] = file_path

        for k, v in list(data.items()):
            # We only want to inject metadata into things that will become YamlModel.
            # These are: WorkflowModel (root), Tasks, Servers, PythonFunctions.
            # data_types is a dict[str, dict] where values are raw JSON schemas.
            # inputs/output in tasks are also tricky.
            if k in ("data_types", "env", "properties", "inputs", "output", "when"):
                if isinstance(v, dict):
                    # For these, we remove metadata recursively
                    remove_metadata(v)
                continue
            data[k] = inject_metadata(v, file_path)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = inject_metadata(data[i], file_path)
    return data


def remove_metadata(data: Any) -> Any:
    if isinstance(data, dict):
        data.pop("__line__", None)
        data.pop("__file_path__", None)
        for v in data.values():
            remove_metadata(v)
    elif isinstance(data, list):
        for item in data:
            remove_metadata(item)
    return data


def format_yaml_error(
    message: str,
    line_number: Optional[int],
    file_path: Optional[str],
    yaml_content: Optional[str] = None,
) -> str:
    parts = []
    location = ""
    if file_path:
        location = f'File "{file_path}"'
    if line_number:
        if location:
            location += f", line {line_number}"
        else:
            location = f"Line {line_number}"

    if location:
        parts.append(f"Error at {location}:")

    parts.append(message)

    if line_number and yaml_content:
        lines = yaml_content.splitlines()
        start = max(0, line_number - 3)
        end = min(len(lines), line_number + 2)
        snippet = []
        for i in range(start, end):
            l_num = i + 1
            prefix = "--> " if l_num == line_number else "    "
            snippet.append(f"{prefix}{l_num:4} | {lines[i]}")
        parts.append("\n" + "\n".join(snippet))

    return "\n".join(parts)


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
                from kavalai.functionkernel import pythontool

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
    def from_yaml_path(cls, yaml_path: str):
        with open(yaml_path, "r") as f:
            yaml_string = f.read()
            try:
                data = yaml.load(yaml_string, Loader=LineLoader)  # nosec B506
                inject_metadata(data, file_path=yaml_path)
                workflow_model = WorkflowModel(**data)
                return cls(workflow_model, yaml_content=yaml_string)
            except ValidationError as e:
                raise WorkflowException(f"Workflow validation failed: {e}") from e

    @classmethod
    def from_yaml(cls, yaml_string: str):
        try:
            data = yaml.load(yaml_string, Loader=LineLoader)  # nosec B506
            inject_metadata(data)
            workflow_model = WorkflowModel(**data)
            return cls(workflow_model, yaml_content=yaml_string)
        except ValidationError as e:
            raise WorkflowException(f"Workflow validation failed: {e}") from e

    def get_data_type(self, name: str) -> Type[BaseModel]:
        """Retrieve a generated Pydantic model by name."""
        if name not in self.models:
            raise KeyError(f"Data type '{name}' was not defined in YAML datatypes.")
        return self.models[name]

    async def run_llm_task(
        self,
        task: LLMTask,
        run_context: RunContext,
        queue: asyncio.Queue | None,
    ):
        input_data = {}
        for name, info in task.inputs.items():
            if info.value is None and info.name is None:
                info = info.model_copy(update={"value": name})
            input_data[name] = await run_context.resolve_input_info(info)

        # Render prompt with templates and context
        try:
            rendered_prompt = await run_context.render_prompt(task.prompt)
        except ValueError as e:
            raise WorkflowException(
                format_yaml_error(
                    str(e),
                    task.line_number,
                    task.file_path,
                    self.yaml_content,
                )
            ) from e
        input_text = make_prompt(rendered_prompt, input_data)

        system_message = dict(role="system", content=input_text)
        messages = [system_message]

        if task.use_history and self.agent_service and run_context.session_id:
            history = await self.agent_service.get_chat_history(run_context.session_id)
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
        if task.stream_output and queue is not None:
            streamer = Streamer(task.output, queue)

        client = LLMClient(model=llm_model)
        response, stats = await client.chat_completions(
            response_model=self.get_data_type(task.output),
            messages=messages,
            streamer=streamer,
            temperature=temperature,
        )
        if self.agent_service:
            await self.agent_service.add_model_call_stats(
                stats=stats, agent_id=run_context.agent_id
            )

        logger.info(f"Setting {task.output} = {response}")
        run_context.data[task.output] = response

        if self.agent_service and run_context.run_id:
            await self.agent_service.add_task(
                agent_id=run_context.agent_id,
                session_id=run_context.session_id,
                run_id=run_context.run_id,
                name=task.name,
                inputs={"prompt": input_text},
                output=response.model_dump()
                if isinstance(response, BaseModel)
                else {"result": response},
            )

    async def run_rest_tool(
        self,
        task: RestTask,
        run_context: RunContext,
        queue: asyncio.Queue | None,
    ):
        inputs = await run_context.prepare_tool_inputs(task)

        # Use FunctionKernel to call REST tool
        tool_uri = f"rest://{task.rest_server}.{task.tool}"
        output_type = self.get_data_type(task.output)

        result = await self.kernel.call_tool(
            tool_uri=tool_uri,
            arguments=inputs,
            output_type=output_type,
            method=task.method,
        )

        debug_data = str(result)[:50]
        logger.info(f"Setting {task.output} = {debug_data}")
        run_context.data[task.output] = result

        # Publish to stream
        if task.stream_output and queue is not None:
            streamer = Streamer(task.output, queue)
            stream_value = (
                result.model_dump_json()
                if isinstance(result, BaseModel)
                else str(result)
            )
            await streamer.stream_complete(stream_value)

        # Store the tool run info.
        if self.agent_service and run_context.run_id:
            await self.agent_service.add_task(
                agent_id=run_context.agent_id,
                session_id=run_context.session_id,
                run_id=run_context.run_id,
                name=task.name,
                inputs={"tool": task.tool, "arguments": inputs},
                output=result.model_dump() if isinstance(result, BaseModel) else result,
            )

    async def run_mcp_tool(
        self, task: McpTask, run_context: RunContext, queue: asyncio.Queue | None
    ):
        inputs = await run_context.prepare_tool_inputs(task)

        # Use FunctionKernel to call MCP tool
        tool_uri = f"mcp://{task.mcp_server}.{task.tool}"
        output_type = self.get_data_type(task.output)

        result = await self.kernel.call_tool(
            tool_uri=tool_uri,
            arguments=inputs,
            output_type=output_type,
        )

        debug_data = str(result)[:50]
        logger.info(f"Setting {task.output} = {debug_data}")
        run_context.data[task.output] = result

        # Publish to stream
        if task.stream_output and queue is not None:
            streamer = Streamer(task.output, queue)
            stream_value = (
                result.model_dump_json()
                if isinstance(result, BaseModel)
                else str(result)
            )
            await streamer.stream_complete(stream_value)

        # Store the tool run info.
        if self.agent_service and run_context.run_id:
            await self.agent_service.add_task(
                agent_id=run_context.agent_id,
                session_id=run_context.session_id,
                run_id=run_context.run_id,
                name=task.name,
                inputs={
                    "mcp_server": task.mcp_server,
                    "tool": task.tool,
                    "arguments": inputs,
                },
                output=result.model_dump() if isinstance(result, BaseModel) else result,
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

        result = await self.kernel.call_tool(
            tool_uri=tool_uri,
            arguments=inputs,
            output_type=output_type,
        )

        if task.output:
            run_context.data[task.output] = result

            if task.stream_output and queue is not None:
                streamer = Streamer(task.output, queue)
                await streamer.stream_complete(
                    result.model_dump_json()
                    if hasattr(result, "model_dump_json")
                    else str(result)
                )

        # Record in DB
        if self.agent_service and run_context.run_id:
            await self.agent_service.add_task(
                agent_id=run_context.agent_id,
                session_id=run_context.session_id,
                run_id=run_context.run_id,
                name=task.name,
                inputs={
                    "python_tool": task.python_tool,
                    "arguments": inputs,
                },
                output=to_plain(result),
            )

    async def run_combine(
        self, task: CombineTask, run_context: RunContext, queue: asyncio.Queue | None
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
            if task.stream_output and queue is not None:
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
            if task.stream_output and queue is not None:
                streamer = Streamer(task.output, queue)
                await streamer.stream_complete(
                    run_context.data[task.output].model_dump_json()
                )
            logger.info(f"Combined inputs into {task.output}")

    async def run_planning_agent(
        self, task: AgentTask, run_context: RunContext, queue: asyncio.Queue | None
    ):
        """Invoke the PlanningAgent for complex multi-step tasks."""
        # 1. Prepare tool inputs
        input_data = await run_context.prepare_tool_inputs(task)

        # 2. Get response model
        response_model = self.get_data_type(task.output)

        # 3. Setup streamer
        streamer = None
        if task.stream_output and queue is not None:
            streamer = Streamer(task.output, queue)

        # 4. LLM client
        llm_model = self.workflow_model.llm_model or self.env.str(
            "KAVALAI_DEFAULT_LLM_MODEL"
        )
        llm_client = LLMClient(model=llm_model)

        # 5. Initialize PlanningAgent
        temperature = (
            task.temperature
            if task.temperature is not None
            else self.workflow_model.temperature
        )
        planning_agent = PlanningAgent(
            kernel=self.kernel,
            run_context=run_context,
            llm_client=llm_client,
            input_data=input_data,
            response_model=response_model,
            streamer=streamer,
            temperature=temperature,
            stream_updates=task.stream_updates,
            stream_output=task.stream_output,
        )

        # 6. Fetch chat history
        chat_history = []
        if task.use_history and self.agent_service and run_context.session_id:
            history = await self.agent_service.get_chat_history(run_context.session_id)
            for msg in history:
                chat_history.append({"role": msg.role, "content": msg.content})

        # 7. Run PlanningAgent
        rendered_prompt = None
        if task.prompt:
            try:
                rendered_prompt = await run_context.render_prompt(task.prompt)
            except ValueError as e:
                raise WorkflowException(
                    format_yaml_error(
                        str(e),
                        task.line_number,
                        task.file_path,
                        self.yaml_content,
                    )
                ) from e

        result = await planning_agent.run(
            task=rendered_prompt,
            chat_history=chat_history,
            max_iterations=task.max_steps,
        )

        # 8. Store results
        run_context.data[task.name] = result
        run_context.data[task.output] = result

    async def run_rag_task(
        self,
        task: RagQueryTask,
        run_context: RunContext,
        queue: asyncio.Queue | None,
    ):
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

        if not text:
            run_context.data[task.name] = []
        else:
            # 1. Get embedding model from workflow
            model = self.workflow_model.embedding_model

            # 2. Initialize RagService
            rag_service = RagService.from_session_maker(
                self.agent_service.session_maker, model
            )
            results = await rag_service.query(
                text=text,
                top_k=task.top_k,
                collection_name=task.collection_name,
                source_ids=task.source_ids,
                keep_best=task.keep_best,
            )

            # 3. Store results in run_context.data (only similarity, content, and source_id)
            run_context.data[task.name] = [
                {
                    "similarity": r.similarity,
                    "content": r.content,
                    "source_id": r.source_id,
                }
                for r in results
            ]

        # 4. Handle output mapping if defined
        if task.output:
            run_context.data[task.output] = run_context.data[task.name]

        # 5. Handle streaming (optional for RAG, usually just one completion event)
        if task.stream_output and queue:
            from kavalai.llm_clients.common import StreamContent

            await queue.put(
                StreamContent(
                    role="assistant",
                    content="",
                    name=task.name,
                    task_type="rag_query",
                    complete=True,
                )
            )

        # Store the tool run info.
        if self.agent_service and run_context.run_id:
            await self.agent_service.add_task(
                agent_id=run_context.agent_id,
                session_id=run_context.session_id,
                run_id=run_context.run_id,
                name=task.name,
                inputs={
                    "text": text,
                    "top_k": task.top_k,
                    "collection_name": task.collection_name,
                },
                output=run_context.data[task.name],
            )

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
        try:
            parsed_input = self.get_data_type("input")(**input_data)
        except ValidationError as e:
            input_info = self.workflow_model.data_types.get("input", {})
            line_number = input_info.get("__line__")
            file_path = input_info.get("__file_path__")
            raise WorkflowException(
                format_yaml_error(
                    f"Validation error for 'input' data type: {e}",
                    line_number,
                    file_path,
                    self.yaml_content,
                )
            ) from e

        run_context = RunContext(agent_service=agent_service)
        run_context.data["input"] = parsed_input
        run_context.templates = {t.name: t.value for t in self.workflow_model.templates}

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
        try:
            streamer = Streamer(name="workflow", queue=queue) if queue else None
            for task in self.workflow_model.tasks:
                if streamer and task.stream_updates:
                    await streamer.stream_complete(task.name, name="running_task")

                if task.when:
                    if not await run_context.evaluate_condition(task.when):
                        logger.info(f"Skipping task <{task.name}> due to condition")
                        continue

                logger.info(f"Running task {task.name}")
                try:
                    if isinstance(task, AgentTask):
                        await self.run_planning_agent(task, run_context, queue)
                    elif isinstance(task, LLMTask):
                        await self.run_llm_task(task, run_context, queue)
                    elif isinstance(task, McpTask):
                        await self.run_mcp_tool(task, run_context, queue)
                    elif isinstance(task, PythonTask):
                        await self.run_python_tool(task, run_context, queue)
                    elif isinstance(task, RestTask):
                        await self.run_rest_tool(task, run_context, queue)
                    elif isinstance(task, CombineTask):
                        await self.run_combine(task, run_context, queue)
                    elif isinstance(task, RagQueryTask):
                        await self.run_rag_task(task, run_context, queue)
                    else:
                        logger.warning(f"Unknown task type: {type(task)}")
                except Exception as e:
                    if isinstance(e, WorkflowException):
                        raise e
                    raise WorkflowException(
                        format_yaml_error(
                            f"Error in task '{task.name}': {e}",
                            task.line_number,
                            task.file_path,
                            self.yaml_content,
                        )
                    ) from e

                if task.stop:
                    logger.info(
                        f"Stopping workflow after task <{task.name}> due to stop: True"
                    )
                    break
        finally:
            # Cleanup MCP sessions/clients using FunctionKernel
            await self.kernel.close()

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
