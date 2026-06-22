# Kaval.AI — Claude Code Guide

## Project Overview

Kaval.AI is a YAML-based AI agent framework with two main components:
- **kavalai.agents**: Core SDK/runtime running on client infrastructure (agent logic, workflows, own DB).
- **kavalai.backoffice**: Management UI for configuring and monitoring agents.

Uses `loguru` for all logging and prefers f-strings for formatting.

## Running code and tests.

Use the `uv` and `.venv` virtual environments to run code and tests — the
project `.venv` is the same interpreter PyCharm/Junie use by default
(`.venv/bin/python -m pytest ...`).

PyCharm/Junie run configurations also load the project `.env` file (via the
EnvFile/built-in dotenv support), which supplies secrets and config such as
`OPENAI_API_KEY`, `GEMINI_API_KEY`, `SERPER_API_KEY`, and the `KAVALAI_*` DB
settings. `conftest.py` does **not** auto-load `.env`, so when running from the
shell you must load it yourself to match the PyCharm environment — otherwise
integration tests gated on these keys silently skip:

```bash
# Load .env, then run tests with the project venv (mirrors PyCharm/Junie)
set -a && source .env && set +a
.venv/bin/python -m pytest

# Equivalently, with uv:
uv run --env-file .env pytest
```

## Key Directories

| Path | Purpose |
|------|---------|
| `kavalai/agents/` | Core SDK: workflow engine, planning agent, sessions, RAG |
| `kavalai/llm_clients/` | Native LLM clients (OpenAI, Gemini, Ollama) + v2 variants |
| `kavalai/prices/` | LLM pricing data and cost calculation |
| `kavalai/tools/` | Utility tools (browser, RSS, web search, HTTP) |
| `kavalai/sql_migrations/` | SQL migrations for `app` (agents) and `backoffice` schemas |
| `backoffice/` | FastAPI backoffice API + DB + project service |
| `frontend/` | Angular UI (Tailwind CSS + DaisyUI) |
| `tests/` | Pytest backend test suite |
| `examples/` | Runnable usage examples |
| `scripts/` | DB migration tool (`migrate_db.py`) |

## Key Files

- `kavalai/agents/workflow.py` — core workflow engine (YAML → execution)
- `kavalai/agents/planning_agent.py` — multi-step LLM planner with tool calling
- `kavalai/agents/agent.py` — modular planning agent (uses the native LLM clients)
- `kavalai/functionkernel.py` — tool registration and execution (REST, MCP, Python)
- `kavalai/agents/workflow_model.py` — Pydantic data models for workflows
- `kavalai/agents/workflow_validation.py` — workflow validation logic
- `kavalai/agents/run_context.py` — RunContext model and context resolution helpers
- `kavalai/workflow/render.py` — renders a workflow graph to an SVG diagram (`render_workflow_svg`); used by the docs build (`docs/_ext/workflow_svgs.py`) and the backoffice `POST /workflows/render-svg` endpoint. The frontend `workflow-graph` component displays that backend SVG.
- `backoffice/server.py` — FastAPI backoffice API (agents, sessions, stats, projects)
- `kavalai/agents/server.py` — Agent REST server (sync + SSE streaming)
- `kavalai/llm_clients/llm_client.py` — high-level LLM client interface

## Backend Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=kavalai tests/

# Run specific test file
pytest tests/test_functionkernel.py
```

- Framework: `pytest` + `pytest-asyncio`
- Target: **100% coverage** for new and modified code
- Keep tests for a single source file in a single test file (e.g. `agent.py` → `test_agent.py`)
- At the end of every coding task, update the relevant tests and run them to catch regressions and confirm new behavior; always run tests before submitting a task. Run only failing tests to save time when you know which ones fail
- Mock MCP servers used in tests live in `tests/helpers/`

## Frontend Testing

```bash
cd frontend && npm test -- --watch=false
cd frontend && npm test -- --watch=false --code-coverage
```

- Framework: Jasmine + Karma
- Update mocks when service interfaces change

## Coding Guidelines

- Use modern Angular control flow syntax: `@if`, `@for` — never `*ngIf` / `*ngFor`
- Prefer styles and components from `common.css`; use Tailwind CSS + DaisyUI for styling
- Refactor code blocks with distinct responsibilities into dedicated functions
- Do not update `README.md`
- Python tools must be decorated with `@kavalai.pythontool` and registered via `register_python_tool`
- Keep this `CLAUDE.md` current: after a task, update it for important changes to project structure, components, or workflow

## Dangerous Commands

When the user denies a delete/remove or other destructive command, ask them to perform it manually rather than retrying.

## Tech Stack

- **Backend**: Python, FastAPI, SQLAlchemy (Async), Pydantic
- **Frontend**: Angular, Tailwind CSS, DaisyUI
- **Database**: PostgreSQL (multiple schemas)
- **Workflows**: YAML definitions processed by `Workflow` class
- **Security**: `ruff`, `yamllint`, `bandit`, `gitleaks` pre-commit hooks

## Docker Commands

```bash
# Backoffice
docker run ... backoffice-migrations
docker run ... backoffice-server   # needs KAVALAI_BO_DB_URI, KAVALAI_BO_DB_SCHEMA

# Agent server
docker run ... agent-server        # needs WORKFLOW_YAML_PATH, KAVALAI_DB_URI, KAVALAI_DB_SCHEMA

# Migrations
docker run ... agent-migrations
```
