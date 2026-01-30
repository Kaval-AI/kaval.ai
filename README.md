# Kaval.AI agents

[Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0)

Kaval.AI is a Python SDK for writing AI agents like chatbots.

## Socrates chatbot example

**socrates.yaml**
```yaml
name: Socrates
description: A chatbot that acts and talks like the philosopher Socrates.
llm_profile_name: openai
data_types:
  input:
    type: object
    properties:
      user_message:
        type: string
  output:
    type: object
    properties:
      agent_response:
        type: string
tasks:
  - name: Compute the response
    prompt: "You are Socrates, the ancient Greek philosopher."
    inputs:
      input:
        type: context
        value: input
    output: output
```

### Running the Chatbot Server

To run the Socrates chatbot server, ensure you have set the required environment variables. You can provide them in a `.env` file or directly in the shell:

```bash
export KAVALAI_AGENT_WORKFLOW_PATH=kavalai/demo_agents/socrates.yaml
export KAVALAI_DB_URI=postgresql://user:password@localhost:5432/dbname
export KAVALAI_DB_SCHEMA=agents
export KAVALAI_AGENT_BASIC_AUTH_USER=admin
export KAVALAI_AGENT_BASIC_AUTH_USER_PASSWORD=password
export OPENAI_API_KEY=your_api_key_here

python -m kavalai.agents.server
```

### Talking to the Chatbot via CLI

While the server is running, you can use the CLI chat tool to talk to Socrates:

```bash
python -m kavalai.tools.cli_chat --url http://localhost --port 10000 --user admin --password password
```


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
| `KAVALAI_AGENT_BASIC_AUTH_USER_PASSWORD` | Password for Basic Authentication. |
| `KAVALAI_AGENT_HOST` | Host to bind the server to (default: `0.0.0.0`). |
| `KAVALAI_AGENT_PORT` | Port to bind the server to (default: `10000`). |
| `KAVALAI_SQL_ECHO` | Enable SQL query logging (default: `0`). |

### Example Command

```bash
KAVALAI_AGENT_WORKFLOW_PATH=kavalai/demo_agents/socrates.yaml \
KAVALAI_DB_URI=postgresql://user:password@localhost:5432/dbname \
KAVALAI_DB_SCHEMA=agents \
KAVALAI_AGENT_BASIC_AUTH_USER=admin \
KAVALAI_AGENT_BASIC_AUTH_USER_PASSWORD=password \
OPENAI_API_KEY=your_api_key_here \
python -m kavalai.agents.server
```

## Persona simulation

Simulate a conversation with the agent using persona simulator.
Give YAML definitions of a persona and a task they are trying to accomplish using the agent's help.
