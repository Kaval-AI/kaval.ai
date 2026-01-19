# Junie's Project Analysis: Kaval.AI

This document serves as a guide for Junie (AI Agent) to understand the Kaval.AI project structure, components, and workflow.

## Project Overview
Kaval.AI is an AI agent writing framework where agent steps are defined using YAML. It consists of two main parts:
- **kavalai.agents**: The core SDK/runtime that runs on client infrastructure. It handles agent logic, workflows, and its own database.
- **kavalai.backoffice**: A management UI for configuring and monitoring agents.

## Directory Structure
- `kavalai/`: Main Python package.
    - `agents/`: SDK and agent runtime logic.
        - `workflow.py`: Core workflow execution engine (YAML to execution).
        - `agent_service.py`: Service for managing agent state, sessions, runs, and LLM profiles in the DB.
        - `stats.py`: Statistics and analytics for agents (sessions, runs, messages).
        - `sessions.py`: Service for querying session summaries and metadata.
        - `schema_parser.py`: Pydantic model generation from JSON schemas for input/output validation.
        - `db.py`: Database models for agents, sessions, runs, tasks, messages, embedding profiles, and RAG index.
    - `llm_clients/`: Native LLM client implementations.
        - `common.py`: Common LLM client utilities. Includes `chat_completion_with_stats` for executing LLM calls with comprehensive metric collection (tokens, duration, request/response data, cost) and `compute_embeddings` for generating text embeddings. Refactored to use native OpenAI and Gemini clients instead of instructor for better structured output and stats collection. Includes `get_llm_client` factory.
        - `openai.py`: Native OpenAI client wrapper. Supports structured outputs via `beta.chat.completions.parse`.
        - `gemini.py`: Native Gemini client wrapper. Supports structured outputs via `response_schema` (with schema cleanup for compatibility).
- `prices/`: Pricing data for different LLM providers.
        - `common.py`: Unified Pydantic models (`ModelPricing`, `TokenPricing`) for LLM pricing and cost calculation logic.
        - `openai.py`: OpenAI pricing data using the unified models.
        - `gemini.py`: Gemini pricing data using the unified models.
    - `backoffice/`: API and logic for the management UI.
        - `server.py`: FastAPI server for the backoffice API. Includes endpoints for agents, sessions, stats, projects (including membership management), and workflow visualization.
        - `db.py`: Backoffice-specific DB models (users, projects, memberships).
        - `project_service.py`: Service for managing project-related data and membership.
    - `tools/`: Utility tools (e.g., RSS, OpenAPI parser).
    - `crud.py`: Shared database utility functions.
- `frontend/`: Angular-based project for the backoffice UI.
    - `src/app/models/`: TypeScript interfaces and data models.
        - `agent.ts`, `session.ts`, `chat-message.ts`, `llm-config.ts`, `llm-call-stat.ts`, `project.ts`, `user-details.ts`.
    - `src/app/services/`: Angular services for API interaction and state management.
        - `agent-service.ts`: Handles agent-related API calls.
        - `user-service.ts`: Manages user authentication and profiles.
        - `project-service.ts`: Manages project context and memberships.
    - `src/app/components/`: UI components organized by feature.
        - `agents-page/`, `projects-page/`, `users-page/`: CRUD interfaces for main entities.
        - `conversations-page/`, `session-detail-page/`: Agent session monitoring and debugging.
        - `configs-page/`: LLM profile and provider configuration.
        - `llm-call-stats-page/`: Detailed list of LLM calls with request/response data (paginated).
        - `metrics-page/`: Token usage and cost analytics.
        - `rag-page/`: RAG-related configuration and testing.
        - `sidebar-menu/`, `header/`, `dropdown-menu/`: Layout and navigation components.
        - `json-tree/`: Tree-like component for displaying nested JSON data (IDE-style).
    - `src/styles/`: Global CSS styles and theme definitions.
- `tests/`: Comprehensive backend test suite.
    - `agents/`: Tests for core SDK, workflow execution, and agent database.
    - `backoffice/`: Tests for management API, project isolation, and memberships.
    - `llm_clients/`: Integration and unit tests for LLM providers (OpenAI, Gemini) and embeddings.
    - `prices/`: Tests for pricing models and cost calculations.
    - `tools/`: Tests for utility tools (RSS, OpenAPI parser).
    - `test_persona_simulator.py`: Tests for the persona simulation logic.
- `sql_migrations/`: SQL migration files for both `app` (agents) and `backoffice`.
    - `app/V001__llm_profiles_and_stats.sql`: Defines `llm_profiles` (with `api_key`, `base_url`, `default_mode`) and `llm_call_stats`.
    - `app/V003__rag_system.sql`: Defines `embedding_profiles` and `rag_index` for the RAG system.
- `llm_profiles/`: Example YAML configurations for different LLM providers (OpenAI, Gemini, Anthropic, Azure, Ollama).
- `scripts/`: Utility scripts (e.g., DB migration).
- `demo_agents/`, `demo_tasks/`, `personas/`: Sample configurations and data.

## Key Technical Details
- **Backend**: Python with FastAPI, SQLAlchemy (Async), Pydantic.
- **Frontend**: Angular.
- **Database**: PostgreSQL (multiple schemas/databases).
- **Workflows**: Defined in YAML, processed by `Workflow` class in `kavalai/agents/workflow.py`.
- **Project Isolation**: The backoffice manages multiple "projects". Each project can point to a different agent database.

## Workflow Execution
1. A YAML definition is loaded.
2. `SchemaParser` generates Pydantic models for input/output.
3. `Workflow.run()` executes the steps (prompts, tools).
4. `AgentService` records everything in the agent database.

## Backoffice Security
- Role-based access (Owner, Viewer).
- Project membership verification on API calls.
- Protection against removing or demoting the last owner of a project.
- Authentication via Google OAuth (configured in `server.py`).

## Important Files for Reference
- `kavalai/backoffice/server.py`: API endpoint definitions (Projects, Users, Agents, Stats).
- `kavalai/agents/workflow.py`: Core agent logic.
- `kavalai/agents/db.py`: Agent-side data schema.
- `kavalai/backoffice/db.py`: Backoffice-side data schema (User, Project, ProjectMembership).
- `frontend/src/app/services/`: Angular services (UserService, ProjectService, etc.).
- `frontend/src/app/components/`: Angular components (UsersPage, ProjectsPage, etc.).

## Backend Testing (Python)
- **Framework**: `pytest`
- **Location**: `tests/`
- **Command**: `pytest`
- **Coverage Command**: `pytest --cov=kavalai tests/`
- **Notes**: Backend tests use `pytest-asyncio` for async database operations.

## Frontend Testing (Angular)
- **Framework**: `Jasmine` + `Karma`
- **Location**: `frontend/src/app/**/*.spec.ts`
- **Command**: `cd frontend && npm test -- --watch=false`
- **Coverage Command**: `cd frontend && npm test -- --watch=false --code-coverage`
- **Notes**: Ensure all mocks are updated if service interfaces change.
