# Kaval.AI — Claude Code Guide

## Project Overview

Kaval.AI is a YAML-based AI agent framework with two main components:
- **kavalai.agents**: Core SDK/runtime running on client infrastructure (agent logic, workflows, own DB).
- **kavalai.backoffice**: Management UI for configuring and monitoring agents.

Uses `loguru` for all logging and prefers f-strings for formatting.

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
- `kavalai/agents/v2/agent.py` — v2 planning agent (modular, uses v2 LLM clients)
- `kavalai/functionkernel.py` — tool registration and execution (REST, MCP, Python)
- `kavalai/agents/workflow_model.py` — Pydantic data models for workflows
- `kavalai/agents/workflow_validation.py` — workflow validation logic
- `kavalai/agents/run_context.py` — RunContext model and context resolution helpers
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
- Run relevant tests after every coding task; run only failing tests to save time when you know which ones fail
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
