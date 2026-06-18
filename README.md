<img src="frontend/public/assets/images/iconlogo.svg" alt="Kaval.AI Logo" width="400" height="100"/>

[![CI](https://github.com/Kaval-AI/kaval.ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Kaval-AI/kaval.ai/actions/workflows/ci.yml)

Kaval.AI is an open source Python SDK for writing AI agents and workflow automation pipelines.
Suitable for conversational agents and generic workflow automation.

Features:
- Makes implementing simple workflows simple via YAML-defined workflows.
- Built-in debugging and monitoring tools via modern UI.
- Built-in support for Retrieval augmented generation (RAG) index.
- Agent server REST API with authentication and streaming support.
- Supports calling REST servers with basic authentication.

## YAML Workflow Tutorial

Kaval.AI workflows are defined in YAML. A workflow consists of metadata, data type definitions, server configurations, and a sequence of tasks.

### Basic Structure

```yaml
name: My Agent
description: A brief description of what this agent does.
version: "1.0"
temperature: 0.7  # Global LLM temperature (0.0 to 2.0)
llm_model: openai/gpt-4o  # Optional: default LLM model
embedding_model: openai/text-embedding-3-small # Default embedding model
data_types:
  input:
    type: object
    properties:
      query: { type: string }
  output:
    type: object
    properties:
      answer: { type: string }
tasks:
  - name: My Task
    type: llm
    prompt: "Answer this: {{ input.query }}"
    output: output
```

### Servers and Functions

You can define external tools via REST, MCP, or Python functions.

```yaml
rest_servers:
  - name: my_api
    url: https://api.example.com
    # Or use env vars: url_env: MY_API_URL

mcp_servers:
  - name: sqlite_mcp
    command: npx
    args: ["-y", "@modelcontextprotocol/server-sqlite", "--db", "test.db"]

python_functions:
  - name: my_custom_tool
    path: my_module.my_function

templates:
  - name: greeting
    value: "Hello, {{ name }}!"
```

### Task Types

Kaval.AI supports several task types to build complex logic.

#### 1. LLM Task (`type: llm`)
Executes a prompt using an LLM.
- `prompt`: The prompt to execute (supports Jinja2-like templates).
- `use_history`: Whether to include chat history (default: `true`).

#### 2. Planning Agent Task (`type: agent`)
A multi-step agent that can use tools (REST, MCP, Python) to achieve a goal.
- `max_steps`: Maximum number of iterations (default: `1`).

Example:
```yaml
tasks:
  - name: Research Task
    type: agent
    max_steps: 5
    prompt: "Find the current stock price of AAPL and its 52-week high."
    output: stock_report

data_types:
  stock_report:
    type: object
    properties:
      current_price: { type: number }
      high_52_week: { type: number }
      summary: { type: string }
```
In this example, if the agent calls a tool with `call_id: current_price`, the result will be automatically placed into the `current_price` field of the `stock_report` output.

#### 3. REST Task (`type: rest`)
Calls a REST API endpoint.
- `rest_server`: Name of the defined REST server.
- `tool`: The endpoint path.
- `method`: HTTP method (default: `get`).

#### 4. MCP Task (`type: mcp`)
Calls a tool from an MCP server.
- `mcp_server`: Name of the defined MCP server.
- `tool`: Tool name.

#### 5. Python Task (`type: python`)
Executes a registered Python function.
- `python_tool`: Name of the defined Python function.

#### 6. RAG Query Task (`type: rag_query`)
Performs a semantic search in the RAG index.
- `text`: The query text.
- `top_k`: Number of results to return (default: `5`).

#### 7. Combine Task (`type: combine`)
Combines multiple inputs into a structured output without an LLM call.

### Task Control & Context

- `inputs`: Map of inputs for the task. Each input can be a `literal`, `context` (current run), or `history` (previous runs).
- `when`: Conditional execution using operators like `eq`, `gt`, `contains`, `all`, `any`, etc.
- `stream_updates`: Stream progress updates (e.g., thoughts).
- `stream_output`: Stream the final task output.

Example with conditions and inputs:
```yaml
  - name: Conditional Task
    type: llm
    when: { gt: ["{{ input.score }}", 0.5] }
    inputs:
      topic: { type: context, value: "previous_task_name.result_field" }
    prompt: "Discuss {{ topic }}"
    output: output
```

## License
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



## News Summarizer agent example

**news_summarizer.yaml**
```yaml
name: News Summarizer
description: An agent that summarizes news from an RSS feed.
llm_profile_name: openai
data_types:
  input:
    type: object
    properties:
      rss_url: { type: string }
  news_feed:
    type: object
    properties:
      title: { type: string }
      items:
        type: array
        items:
          type: object
          properties:
            title: { type: string }
            summary: { type: string }
  output:
    type: object
    properties:
      agent_response: { type: string }
rest_servers:
  - name: rss_api
    url: http://localhost:10001
tasks:
  - name: Fetch news
    tool: /get_rss_feed
    rest_server: rss_api
    inputs:
      url: { type: context, value: input.rss_url }
    output: news_feed
  - name: Summarize news
    prompt: "Summarize the following news items into a concise report."
    inputs:
      news: { type: context, value: news_feed.items }
    output: output
```

### Running the Agent Server

To run the news summarizer agent server, ensure you have set the required environment variables. You can provide them in a `.env` file or directly in the shell:

```bash
export KAVALAI_AGENT_WORKFLOW_PATH=news_summarizer.yaml
export KAVALAI_DB_URI=postgresql://user:password@localhost:5432/dbname
export KAVALAI_DB_SCHEMA=agents
export KAVALAI_DB_POOL_SIZE=0
export KAVALAI_DB_MAX_OVERFLOW=0
export KAVALAI_AGENT_BASIC_AUTH_USER=admin
export KAVALAI_AGENT_BASIC_AUTH_PASSWORD=password
export OPENAI_API_KEY=your_api_key_here

python -m kavalai.agents.server
```

### Talking to the Agent via CLI

While the server is running, you can use the CLI chat tool to interact with the agent:

```bash
python -m kavalai.tools.cli_chat --url http://localhost --port 10000 --user admin --password password
```

### Agent Server Endpoints

The agent server provides two primary endpoints for executing workflows:

#### Synchronous: `/run_agent`
Returns the final result of the entire workflow in a single JSON response once all tasks are complete.

- **Use case:** Simple interactions where you only need the final output.
- **Payload:**
  ```json
  {
    "session_id": "...",
    "data": { "agent_response": "..." }
  }
  ```

#### Streaming: `/stream_agent`
Returns a stream of Server-Sent Events (SSE) as the workflow executes. This allows for real-time updates, especially for long-running LLM prompts. Each event is a JSON object prefixed with `data: `.

- **Use case:** Chat interfaces and real-time applications where you want to show the agent's response as it's being generated.
- **Payload (Stream):**
  ```json
  {"type": "partial", "name": "output", "value": "{\"agent_response\": \"The\"}"}
  {"type": "partial", "name": "output", "value": "{\"agent_response\": \"The report\"}"}
  {"type": "complete", "name": "output", "value": "{\"agent_response\": \"The report is ready.\"}"}
  ```

**Important:** The `stream: true` keyword in task definitions only has an effect when using the `/stream_agent` endpoint. If you use `/run_agent`, all tasks will execute to completion before the final response is sent.


# Installation

You need access to [Kaval.AI Github repository](https://github.com/Kaval-AI/kaval.ai).
For any questions, please write to *tpetmanson@gmail.com*.

Add the Github repository dependency to `requirements.txt` or your `pyproject.toml`:

```aiignore
kavalai@git+ssh://git@github.com/Kaval-AI/kaval.ai.git@main
```

```
pip install .
pip install .[test]
```

## Database migrations

We use a custom migration script to manage database schemas. You can run migrations using the `kavalai.migrate_db` module.
It supports two types of migrations: `app` (for agents) and `backoffice`.

```bash
python -m kavalai.migrate_db app
python -m kavalai.migrate_db backoffice
```



## Local development

**Backend setup**

- Clone the Kaval.AI repository and go to the directory.
- Create a virtualenv (we recommend using Pycharm to create it automatically).
- Install dependencies in the virtual env `pip install -r requirements.txt`.
- Make a copy of `.env.example` as `.env` and fill in the missing details (ask your team lead for details).
- Install [docker](https://docs.docker.com/engine/install/ubuntu/). You probably have to reboot your computer in the process.
- Run `docker compose up -D` to bring up Postgres database.
- Apply database migrations to local and test databases:
  ```bash
  python -m kavalai.migrate_db app
  python -m kavalai.migrate_db backoffice
  ```


**Frontend setup**

- Install [nvm](https://github.com/nvm-sh/nvm)
- Install latest npm `nvm install [latest version]`
- Install [Angular](https://angular.dev/installation)
- Next, go to the `frontend/` directory.
- Install `npm install -g karma-cli`
- Run `npm install`

# Running agents

To run an agent server, you need to configure the environment variables.

### Required Environment Variables

| Variable | Description |
| --- | --- |
| `KAVALAI_AGENT_WORKFLOW_PATH` | Path to the agent's workflow YAML file. |
| `KAVALAI_DB_URI` | Database connection string (e.g., `postgresql://user:password@localhost/dbname`). |
| `KAVALAI_DB_SCHEMA` | Database schema name (default: `public`). |
| `OPENAI_API_KEY` | Your OpenAI API key (if using OpenAI models). |
| `GEMINI_API_KEY` | Your Gemini API key (if using Gemini models). |

### Optional Environment Variables

| Variable | Description |
| --- | --- |
| `KAVALAI_AGENT_BASIC_AUTH_USER` | Username for Basic Authentication. |
| `KAVALAI_AGENT_BASIC_AUTH_PASSWORD` | Password for Basic Authentication. |
| `KAVALAI_AGENT_HOST` | Host to bind the server to (default: `0.0.0.0`). |
| `KAVALAI_AGENT_PORT` | Port to bind the server to (default: `10000`). |
| `KAVALAI_SQL_ECHO` | Enable SQL query logging (default: `0`). |
| `KAVALAI_DB_POOL_SIZE` | Database connection pool size (default: `0` for no pooling). |
| `KAVALAI_DB_MAX_OVERFLOW` | Database connection pool max overflow (default: `0`). |
| `KAVALAI_OPENAI_SERVICE_TIER` | The service tier to use for OpenAI API calls (e.g., `priority`). |

### Example Command

```bash
KAVALAI_AGENT_WORKFLOW_PATH=kavalai/demo_agents/socrates.yaml \
KAVALAI_DB_URI=postgresql://user:password@localhost:5432/dbname \
KAVALAI_DB_SCHEMA=agents \
KAVALAI_DB_POOL_SIZE=0 \
KAVALAI_DB_MAX_OVERFLOW=0 \
KAVALAI_AGENT_BASIC_AUTH_USER=admin \
KAVALAI_AGENT_BASIC_AUTH_PASSWORD=password \
OPENAI_API_KEY=your_api_key_here \
python -m kavalai.agents.server
```

## Kavalai agent YAML workflow tutorial

Kaval.AI agents are defined using a YAML workflow.
A workflow consists of `data_types` and a list of `tasks`.

### Data types

Kaval.AI uses JSON Schema-like definitions to define data types. These are used to validate inputs and outputs of the workflow and individual tasks.

**Example:**
```yaml
data_types:
  input:
    type: object
    properties:
      rss_url: { type: string }
  news_feed:
    type: object
    properties:
      title: { type: string }
      items:
        type: array
        items:
          type: object
          properties:
            title: { type: string }
            summary: { type: string }
  output:
    type: object
    properties:
      agent_response: { type: string }
```

Supported types include `string`, `integer`, `number`, `boolean`, `array`, and `object`.

#### Using $ref for complex types

You can use `$ref` to reference other data types defined in `data_types`. This is useful for building complex, nested data structures.

**Example:**
```yaml
data_types:
  news_item:
    type: object
    properties:
      title: { type: string }
      summary: { type: string }
  news_feed:
    type: object
    properties:
      title: { type: string }
      items: { type: array, items: { $ref: news_item } }
```

### Tasks

Tasks are the building blocks of a workflow. There are three main types of tasks: `prompt`, `run_tool`, and `combine`.

#### Prompt (LLM call)
A `prompt` task sends a message to an LLM.

- `prompt`: The system message or instructions for the LLM.
- `inputs`: Data from the context to be included in the prompt.
- `output`: The name of the data type the LLM response should conform to.
- `stream`: (Optional) Set to `true` to stream the response. **Note:** Streaming data is only available when using the `/stream_agent` endpoint.

**Example:**
```yaml
  - name: Summarize news
    prompt: "Summarize the following news items."
    inputs:
      news_items: { type: context, value: news_feed.items }
    output: output
    stream: true
```

#### Run Tool (REST call)
A `run_tool` task calls an external REST API.

- `tool`: The API endpoint path.
- `rest_server`: The name of the REST server (defined in `rest_servers`).
- `method`: HTTP method (e.g., `get`, `post`).
- `inputs`: Parameters or body for the API call.
- `output`: The data type the API response should be parsed into.

**Example:**
```yaml
  - name: Fetch news
    tool: /get_rss_feed
    rest_server: rss_api
    method: get
    inputs:
      url: { type: context, value: input.rss_url }
    output: news_feed
```

#### Combine
A `combine` task merges multiple context values into a single data type without calling an LLM or an external tool.

**Basic Example:**
Merging inputs into a named data type.
```yaml
  - name: Finalize news feed
    inputs:
      title: { type: literal, value: "Latest News" }
      items: { type: context, value: fetch_task.items }
    output: news_feed
```

**Special case for final output:**
If the `output` of a `combine` task is a dictionary instead of a string, it maps inputs to the fields of the special `output` data type. This is the standard way to prepare the final response of the workflow.

```yaml
  - name: Finalize output
    inputs:
      agent_response: { type: context, value: summarization_task.text }
    output:
      agent_response: { type: context, value: summarization_task.text }
```

### REST Servers and Authentication

You can define external REST servers in the `rest_servers` section. To avoid hardcoding sensitive information or environment-specific URLs, you can use environment variables.

#### Configuration via Environment Variables

- `url_env`: Environment variable name containing the base URL.
- `username_env`: Environment variable name containing the username for Basic Auth.
- `password_env`: Environment variable name containing the password for Basic Auth.

**Example:**
```yaml
rest_servers:
  - name: rss_api
    url_env: RSS_API_URL
    username_env: RSS_API_USER
    password_env: RSS_API_PASSWORD
```

When using `url_env`, the SDK will look up the value of `RSS_API_URL` in your environment (e.g., in your `.env` file). If `username_env` and `password_env` are provided, the `run_tool` task will automatically use Basic Authentication.

### Context and Inputs

Inputs for tasks can be `literal` or `context` values. `context` values use a dotted path to reference data produced by previous tasks or the initial `input`.

- `{ type: literal, value: "https://news.ycombinator.com/rss" }`
- `{ type: context, value: news_feed.items }`


## Persona simulation

Simulate a conversation with the agent using persona simulator.
Give YAML definitions of a persona and a task they are trying to accomplish using the agent's help.
