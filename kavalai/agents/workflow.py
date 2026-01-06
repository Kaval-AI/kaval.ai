import asyncio
import json
import logging
from typing import List, Dict, Type, Optional, Any
from uuid import UUID

import instructor
import yaml
from mcp import ClientSession
from mcp.client.sse import sse_client
from pydantic import BaseModel

from kavalai.agents.agent_service import AgentService
from kavalai.agents.schema_parser import SchemaParser

logger = logging.getLogger(__name__)


class Task(BaseModel):
    name: str
    inputs: List[str] = []
    output: str
    # LLM call
    prompt: Optional[str] = None
    # MCP tool call
    tool: Optional[str] = None
    mcp_server: Optional[str] = None
    arguments: Optional[dict[str, Any]] = None


class McpServer(BaseModel):
    name: str
    url: str
    username: Optional[str] = None
    password: Optional[str] = None


class WorkflowModel(BaseModel):
    name: str
    description: str
    llm_provider: str
    data_types: dict[str, dict]
    mcp_servers: list[McpServer]
    tasks: list[Task]


class RunContext(BaseModel):
    """Runtime data for a single interaction."""

    agent_id: Optional[UUID] = None
    interaction_id: Optional[UUID] = None
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


class Workflow:
    def __init__(
        self,
        workflow_model: WorkflowModel,
        agent_service: Optional[AgentService] = None,
    ):
        self.workflow_model = workflow_model
        self.agent_service = agent_service

        # Data types used in the workflow.
        self.parser = SchemaParser(workflow_model.data_types)
        self.models: Dict[str, Type[BaseModel]] = self.parser.parse_all()

        # MCP servers
        self.mcp_servers = {
            server.name: server for server in workflow_model.mcp_servers
        }
        self.tasks = {task.name: task for task in workflow_model.tasks}

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, "r") as f:
            workflow_model = WorkflowModel(**yaml.safe_load(f.read()))
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
        client = instructor.from_provider(
            self.workflow_model.llm_provider,
            async_client=True,
            mode=instructor.Mode.JSON,
        )
        # Run the completion.
        system_message = dict(role="system", content=input_text)
        logger.info("SYSTEM: %s", system_message)
        response = await client.chat.completions.create(
            response_model=self.get_data_type(task.output),
            messages=[system_message],
        )
        logger.info(f"Setting {task.output} = {response.model_dump_json()}")
        run_context.data[task.output] = response

    async def run_tool(self, task: Task, run_context: RunContext):
        async with sse_client(self.mcp_servers[task.mcp_server].url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                call_result = await session.call_tool(
                    task.tool, arguments=task.arguments
                )
                result = json.loads(call_result.content[0].text)
                # If we have a data type defined for tool result, convert it.
                if self.get_data_type(task.output):
                    result = self.get_data_type(task.output)(**result)
                logger.info(f"Setting {task.output} = {result.model_dump_json()}")
                run_context.data[task.output] = result

    async def run(
        self, input_data: dict, interaction_id: Optional[UUID] = None
    ) -> BaseModel:
        input_data = self.get_data_type("input")(**input_data)
        run_context = RunContext()
        run_context.data["input"] = input_data

        # Initialize agent and interaction data, if agent_service is defined.
        if self.agent_service:
            agent_id = self.agent_service.get_agent_id(self.workflow_model.name)
            if not agent_id:
                agent_id = self.agent_service.create_agent(self.workflow_model.name)
            run_context.agent_id = agent_id
            run_context.interaction_id = interaction_id
            if not interaction_id:
                run_context.interaction_id = self.agent_service.create_interaction(
                    self.workflow_model.name
                )
            # Add user message to chat history.
            self.agent_service.add_message(
                run_context.interaction_id,
                "user",
                run_context.data["input"].user_message,
            )
        # Run the steps sequentially.
        for task in self.workflow_model.tasks:
            logger.info("Running task <%s>", task.name)
            if task.prompt:
                await self.run_prompt(task, run_context)
            elif task.tool:
                await self.run_tool(task, run_context)
            else:
                raise WorkflowException(str(task))
        # Add agent response to chat history.
        if self.agent_service:
            self.agent_service.add_message(
                run_context.interaction_id,
                "assistant",
                run_context.data["output"].agent_response,
            )
        return run_context.data["output"]

    def __repr__(self):
        return f"<Workflow agent='{self.agent_name}' tasks={len(self.tasks)}>"


async def main():
    workflow = Workflow.from_yaml("kavalai/agents/example.yaml")
    result = await workflow.run(
        dict(user_message="Hey man, what is the meaning of life, dude?")
    )
    logger.info("RESULT: %s", result.model_dump_json())


if __name__ == "__main__":
    asyncio.run(main(), debug=True)
