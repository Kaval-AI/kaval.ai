import asyncio
import logging
from typing import List, Dict, Type

import instructor
import yaml
from pydantic import BaseModel

from kavalai.agents.schema_parser import SchemaParser

logger = logging.getLogger(__name__)


class Task(BaseModel):
    name: str
    prompt: str
    llm_provider: str
    inputs: List[str]
    output: str


class WorkflowModel(BaseModel):
    name: str
    description: str
    data_types: dict[str, dict]
    tasks: list[Task]
    steps: list[str]


class RunContext(dict):
    pass


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
    def __init__(self, workflow_model: WorkflowModel):
        self.workflow_model = workflow_model

        # Data types used in the workflow.
        self.parser = SchemaParser(workflow_model.data_types)
        self.models: Dict[str, Type[BaseModel]] = self.parser.parse_all()

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

    async def run_task(self, task_name, run_context: RunContext):
        task = self.tasks[task_name]
        input_data = {
            input_name: run_context.get(input_name) for input_name in task.inputs
        }
        input_text = make_prompt(task.prompt, input_data)
        client = instructor.from_provider(
            task.llm_provider, async_client=True, mode=instructor.Mode.JSON
        )
        # Run the completion.
        system_message = dict(role="system", content=input_text)
        response = await client.chat.completions.create(
            response_model=self.get_data_type(task.output),
            messages=[system_message],
        )
        logger.info(f"Setting {task.output} = {response.model_dump_json()}")
        run_context[task.output] = response

    async def run(self, input_data: dict) -> BaseModel:
        input_data = self.get_data_type("input")(**input_data)
        run_context = RunContext(input=input_data)

        for step_name in self.workflow_model.steps:
            logger.info("Running task <%s>", step_name)
            await self.run_task(step_name, run_context)

        return run_context["output"]

    def __repr__(self):
        return f"<Workflow agent='{self.agent_name}' steps={len(self.steps)}>"


async def main():
    workflow = Workflow.from_yaml("kavalai/agents/example.yaml")
    result = await workflow.run(
        dict(user_message="Hello! What is the capital of France?")
    )
    logger.info("RESULT: %s", result.model_dump_json())


if __name__ == "__main__":
    asyncio.run(main(), debug=True)
