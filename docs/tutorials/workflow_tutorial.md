# Creating Agents with Workflow and WorkflowModel

Kaval.AI provides a flexible way to define and run AI agents using YAML or Python-based `WorkflowModel` definitions. This tutorial covers everything from basic programmatic execution to advanced server setups with database integration.

## 1. Core Concepts

At the heart of Kaval.AI are two main components:
- **`WorkflowModel`**: A Pydantic-based model that defines the agent's structure, tasks, and data types.
- **`Workflow`**: The runtime engine that executes a `WorkflowModel`.

### Workflow Structure
A workflow consists of:
- **`name`**: The agent's name.
- **`description`**: What the agent does.
- **`data_types`**: JSON Schema definitions for inputs, outputs, and intermediate states.
- **`tasks`**: A list of execution steps (LLM calls, tool calls, etc.).
- **`mcp_servers`** / **`rest_servers`**: External services the agent can interact with.

---

## 2. Defining a Workflow

You can define a workflow in YAML or directly in Python.

### YAML Definition (`socrates.yaml`)
```yaml
name: Socrates
description: A philosophical AI that answers questions using the Socratic method.
data_types:
  input:
    type: object
    properties:
      question: {type: string}
  output:
    type: object
    properties:
      answer: {type: string}

tasks:
  - name: generate_answer
    prompt: |
      You are Socrates. Answer the following question: {{input.question}}
    output: output
```

### Python Definition
```python
from kavalai.agents.workflow_model import WorkflowModel, LLMTask, TypeInputInfo

workflow_model = WorkflowModel(
    name="Socrates",
    description="A philosophical AI",
    data_types={
        "input": {"type": "object", "properties": {"question": {"type": "string"}}},
        "output": {"type": "object", "properties": {"answer": {"type": "string"}}}
    },
    tasks=[
        LLMTask(
            name="generate_answer",
            prompt="You are Socrates. Answer the following question: {{input.question}}",
            output="output",
            inputs={"question": TypeInputInfo(type="context", value="input.question")}
        )
    ]
)
```

---

## 3. Programmatic Execution

### Simple Execution (No Database)
If you don't need to persist sessions or logs, you can run the workflow directly.

```python
import asyncio
from kavalai.agents.workflow import Workflow

async def main():
    # Load from YAML or use the Python model
    workflow = Workflow.from_yaml_path("socrates.yaml")

    # Run the workflow
    result = await workflow.run(input_data={"question": "What is justice?"})

    print(f"Result: {result.data['answer']}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Execution with Database Integration
To store sessions, runs, and model stats, you need to provide an `AgentService`.

```python
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from kavalai.agents.workflow import Workflow
from kavalai.agents.agent_service import AgentService

async def main():
    # Setup database
    engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/dbname")
    session_factory = async_sessionmaker(engine)

    # Initialize AgentService
    agent_service = AgentService(session_factory)

    # Load workflow
    workflow = Workflow.from_yaml_path("socrates.yaml", agent_service=agent_service)

    # Run with a session ID to persist history
    result = await workflow.run(
        input_data={"question": "What is virtue?"},
        agent_service=agent_service
    )

    print(f"Run ID: {result.run_id}")
    print(f"Answer: {result.data['answer']}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 4. Running the Agent Server

Kaval.AI includes a FastAPI-based server to expose your agent via REST API.

### Simple Server (No Database)
This is useful for testing or lightweight deployments.

```python
import uvicorn
from kavalai.agents.workflow import Workflow
from kavalai.agents.server import create_agent_app

workflow = Workflow.from_yaml_path("socrates.yaml")
app = create_agent_app(workflow)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Server with Database
Provide a `session_provider` to enable persistence in the REST API.

```python
import uvicorn
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from kavalai.agents.workflow import Workflow
from kavalai.agents.server import create_agent_app

# DB Setup
engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/dbname")
session_factory = async_sessionmaker(engine)

# Workflow and App
workflow = Workflow.from_yaml_path("socrates.yaml")
app = create_agent_app(workflow, session_provider=session_factory)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 5. Advanced Configuration

### Using DatabaseManager
For more complex DB setups (pooling, schemas, etc.), use the `DatabaseManager`.

```python
import uvicorn
from kavalai.agents.db import DatabaseManager
from kavalai.agents.workflow import Workflow
from kavalai.agents.server import create_agent_app

db_manager = DatabaseManager()
session_factory = db_manager.get_sessionmaker(
    uri="postgresql+asyncpg://user:pass@localhost/dbname",
    pool_size=20,
    max_overflow=10
)

workflow = Workflow.from_yaml_path("socrates.yaml")
app = create_agent_app(workflow, session_provider=session_factory)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Combined: Programmatic + Server
You can share the same database and workflow logic between a background worker and a REST server.

```python
from kavalai.agents.db import DatabaseManager
from kavalai.agents.agent_service import AgentService
from kavalai.agents.workflow import Workflow
from kavalai.agents.server import create_agent_app

# 1. Shared DB Config
session_factory = DatabaseManager().get_sessionmaker(uri="...")
agent_service = AgentService(session_factory)

# 2. Shared Workflow
workflow = Workflow.from_yaml_path("socrates.yaml", agent_service=agent_service)

# 3. Use in Server
app = create_agent_app(workflow, session_provider=session_factory)

# 4. Use Programmatically (e.g., in a task queue)
async def run_background():
    await workflow.run(input_data={"question": "..."}, agent_service=agent_service)
```

---

## 6. Task Types Reference

Kaval.AI supports several task types in `WorkflowModel`:

| Task Type | Description | Key Fields |
|-----------|-------------|------------|
| `LLMTask` | Standard LLM prompt | `prompt`, `temperature` |
| `RestTask` | Call a REST API tool | `tool`, `rest_server` |
| `McpTask` | Call an MCP tool | `tool`, `mcp_server` |
| `PythonTask`| Run a Python function | `python_tool` |
| `AgentTask` | Orchestrate other tools | `agent_type: "planning"`, `max_steps` |
| `CombineTask`| Merge multiple inputs | No prompt/tool, just `inputs` and `output` |

### Conditional Execution
Every task supports a `when` field for conditional logic:
```yaml
tasks:
  - name: check_sentiment
    prompt: Is this positive? {{input.text}}
    output: sentiment
  - name: send_thank_you
    when: {eq: ["{{sentiment.is_positive}}", true]}
    prompt: Thank the user...
    output: output
```
