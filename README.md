# Kaval.AI Enterprise Agent Management System

Kaval.AI is a Python SDK for developing AI agents that comes with backoffice app that is useful for monitoring and a/b testing different agent configurations like prompts and models.

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

```
HTTP_BASIC_AUTH_USER=user HTTP_BASIC_AUTH_PASSWORD=password python -m kavalai.agents.server demo_agents/silverhand.yaml --port 10000
```

Simulate a conversation with the agent using persona simulator.
Give YAML definitions of a persona and a task they are trying to accomplish using the agent's help.
