# Kaval.AI Enterprise Agent Management System

Kaval.AI is a Python SDK for developing AI agents that comes with backoffice app that is useful for monitoring and a/b testing different agent configurations like prompts and models.

# Installation

You need access to [Kaval.AI Github repository](https://github.com/Kaval-AI/kaval.ai).
For any questions, please write to *tpetmanson@gmail.com*.

Add the Github repository dependency to `requirements.txt`.

```aiignore
kavalai@git+ssh://git@github.com/Kaval-AI/kaval.ai.git@main
```

Run `pip install -r requirements.txt`.

## LLM profiles

LLM profiles define how to connect to various LLM providers. They are YAML files stored in the `llm_profiles/` directory (or passed to the agent server).

**OpenAI example (`llm_profiles/openai.yaml`)**

```yaml
name: openai
provider: openai
model_name: gpt-4o
api_key: your-api-key-here
```

**Gemini example (`llm_profiles/gemini.yaml`)**

```yaml
name: gemini
provider: google
model_name: gemini-1.5-pro
api_key: your-api-key-here
```

## Socrates Chatbot Example

We have included a Socrates chatbot example in `demo_agents/socrates.yaml`. This agent is configured to use the Socratic method in its conversations.

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
    prompt: "You are Socrates, the ancient Greek philosopher. Engage in a Socratic dialogue with the user. Answer their questions with further questions that encourage them to think deeper and examine their own assumptions. Be humble, claim to know nothing, and use irony where appropriate. Your goal is not to provide answers, but to help the user find the truth within themselves."
    inputs:
      input:
        type: context
        value: input
    output: output
```

### Running the Chatbot Server

To run the Socrates chatbot server, use the following command:

```bash
HTTP_BASIC_AUTH_USER=admin HTTP_BASIC_AUTH_PASSWORD=password python -m kavalai.agents.server demo_agents/socrates.yaml --port 10000
```

### Talking to the Chatbot via CLI

While the server is running, you can use the CLI chat tool to talk to Socrates:

```bash
python -m kavalai.tools.cli_chat --url http://localhost --port 10000 --user admin --password password
```



## Local development

**Backend setup**

- Clone the Kaval.AI repository and go to the directory.
- Create a virtualenv (we recommend using Pycharm to create it automatically).
- Install dependencies in the virtual env `pip install -r requirements.txt`.
- Make a copy of `.env.example` as `.env` and fill in the missing details (ask your team lead for details).
- Install [docker](https://docs.docker.com/engine/install/ubuntu/). You probably have to reboot your computer in the process.
- Run `docker compose up -D` to bring up Postgres database.
- Apply database migrations to local and test databases.

```
./scripts/migrate_db.sh scripts/local_db.env
./scripts/migrate_db.sh scripts/test_db.env
```

- Run backend unit tests

```
pytest tests/
```

All of them should pass

**Frontend setup**

- Install [nvm](https://github.com/nvm-sh/nvm)
- Install latest npm `nvm install [latest version]`
- Install [Angular](https://angular.dev/installation)
- Next, go to the `frontend/` directory.
- Install `npm install -g karma-cli`
- Run `npm install`

# Running agents

Runs a demo agent on port 10000 with given access credentials.

```bash
HTTP_BASIC_AUTH_USER=user HTTP_BASIC_AUTH_PASSWORD=password python -m kavalai.agents.server demo_agents/silverhand.yaml --port 10000
```

## Persona simulation

Simulate a conversation with the agent using persona simulator.
Give YAML definitions of a persona and a task they are trying to accomplish using the agent's help.
