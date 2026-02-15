# Junie's Project Analysis: Kaval.AI

This document serves as a guide for Junie (AI Agent) to understand the Kaval.AI project structure, components, and workflow.

## Docker Usage
Kaval.AI can be run using Docker. The provided `Dockerfile` and `entrypoint.sh` support multiple commands:
- `backoffice-migrations`: Run database migrations for the backoffice.
- `agent-migrations`: Run database migrations for the agents.
- `backoffice-server`: Start the backoffice Nginx and FastAPI server (requires `KAVALAI_BO_DB_URI` / `KAVALAI_BO_DB_SCHEMA`).
- `agent-server`: Start an agent REST server (requires `WORKFLOW_YAML_PATH` and `KAVALAI_DB_URI` / `KAVALAI_DB_SCHEMA`).


## Project Overview
Kaval.AI is an AI agent writing framework where agent steps are defined using YAML. It consists of two main parts:
- **kavalai.agents**: The core SDK/runtime that runs on client infrastructure. It handles agent logic, workflows, and its own database.
- **kavalai.backoffice**: A management UI for configuring and monitoring agents.

## Directory Structure
- `kavalai/`: Main Python package.
    - `agents/`: SDK and agent runtime logic.
        - `__init__.py`: Package initialization, defines `PACKAGE_PATH` and `MIGRATIONS_PATH`.
        - `workflow.py`: Core workflow engine (YAML to execution). Supports case-insensitive (snake_case conversion) loading of tasks and workflow models from YAML. `RunContext.resolve_context_value` supports nested dotted paths (e.g., `input.criteria.keywords`). `RunContext.prepare_tool_inputs` always returns a dictionary of inputs, ensuring Pydantic models are dumped to dicts. Tasks without `prompt` or `tool` are "combine" tasks, merging inputs into the specified output. REST servers support both `url` and `url_env` (with `url_env` taking precedence if defined in environment) and use `username_env` and `password_env` for basic auth. Tasks can specify an HTTP `method` for REST tool calls. Supports streaming output for both prompts (incremental) and tool/combine tasks (completion only) via an `asyncio.Queue` passed as the `queue` parameter to `run()`.
        - `workflow_validation.py`: Dedicated module for workflow validation logic, including REST server environment variables and overall workflow structure.
        - `server.py`: REST server for agents. Supports optional HTTP basic authentication. Logs masked database connection details (using `***`), basic auth configuration status, database pooling settings, and OpenAI service tier upon startup. Includes `mask_db_uri` for secure logging of database URIs. Added `/liveness` and `/health` endpoints for K8s probes; `/health` performs a database connectivity check. `create_app_from_env_conf` supports optional parameters to override environment variables. `create_agent_app` stores the workflow in `app.state.workflow`. Supports `/run_agent` for synchronous execution and `/stream_agent` for streaming output (SSE). `stream_agent` also streams a final "output" event containing the full result (session ID and data).
        - `client.py`: Client for interacting with agent servers, handles schema discovery and session management. Now ensures HTTP basic auth is only used if both username and password are provided and non-empty. Added `stream_agent` method for streaming output.
        - `agent_service.py`: Service for managing agent state, sessions, runs, LLM profiles, and embedding profiles in the DB. `get_or_create_agent` updates existing agents if description, schemas or workflow changed. Now includes `total_cost` in LLM and embedding profile views.
        - `rag_service.py`: Service for indexing and querying text using embeddings (RAG). `query` now returns a list of `RagServiceResult` Pydantic models and supports `keep_best` parameter to return only the best match per `source_id`. Now supports `source_ids` parameter to filter results. Added `compute_similarity_matrix` to compute similarities between multiple texts and source IDs in a single query. Added `learn_normalizer` to learn centering and normalization parameters from the index. `RagService` now accepts an optional `normalizer` instance in `__init__` which is used for all embedding computations.
    - `llm_clients/`: Native LLM client implementations.
        - `common.py`: Shared LLM client utilities, including `ModelCallStat` creation, embedding normalization, and `StreamContent` model.
    - `normalizer.py`: Embedding transformation utilities including L1/L2 normalization and centering. Supports learning centering parameters from RAG index, YAML saving/loading (from files or strings), and numpy-optimized batch operations.
        - `llm_client.py`: High-level LLM client interface. Includes `chat_completions` for executing LLM calls with comprehensive metric collection (tokens, duration, request/response data, cost) and `compute_embeddings` for generating text embeddings. Both return a result/stats tuple, letting the caller decide whether to log stats using `_save_model_stats`. Refactored to use native OpenAI and Gemini clients instead of instructor for better structured output and stats collection. Includes `get_llm_client` factory. Supports configurable timeouts via `KAVALAI_LLM_TIMEOUT` (default: 30s). Added streaming support to `chat_completions`.
        - `openai_client.py`: Native OpenAI client wrapper. Supports structured outputs via `beta.chat.completions.parse` and streaming via `beta.chat.completions.stream`. Both `chat_completion` and `compute_embeddings` return a tuple of `(result, ModelCallStat)`. `chat_completion` now includes retry logic (up to 3 attempts) for `LengthFinishReasonError`. Supports `service_tier` parameter (auto, default, flex, priority) for chat completion calls.
        - `gemini_client.py`: Native Gemini client wrapper. Supports structured outputs via `response_schema` (with schema cleanup for compatibility) and streaming via `generate_content_stream`. Both `chat_completions` and `compute_embeddings` return a tuple of `(result, ModelCallStat)`.
    - `prices/`: Pricing data for different LLM providers.
        - `common.py`: Unified Pydantic models (`ModelPricing`, `TokenPricing`) for LLM pricing and cost calculation logic.
        - `openai.py`: OpenAI pricing data using the unified models.
        - `gemini.py`: Gemini pricing data using the unified models.
    - `backoffice/`: API and logic for the management UI.
        - `__init__.py`: Package initialization, defines `PACKAGE_PATH` and `MIGRATIONS_PATH`.
        - `server.py`: FastAPI server for the backoffice API. Includes endpoints for agents, sessions, stats, projects (including membership management), and workflow visualization. `projects_rag_query` supports filtering by `source_ids`.
        - `svg.py`: Utility for generating SVG visualizations of workflows using Graphviz. Supports rendering data nodes with schema properties and resolving reference chains.
        - `db.py`: Backoffice-specific DB models (users, projects, memberships). Includes `DatabaseManager` for handling `postgresql+asyncpg` and `KAVALAI_BO_DB_URI`.
        - `project_service.py`: Service for managing project-related data and membership.
    - `tools/`: Utility tools.
        - `cli_chat.py`: Command line tool for chatting with agents. Now supports Ctrl+D (EOF) to exit and uses streaming server endpoint for real-time responses.
        - `rss.py`: RSS feed parser tool.
        - `openapi_spec_parser.py`: Tool for parsing OpenAPI specifications.
        - `index_csv.py`: Tool for indexing large CSV files into RAG.
    - `crud.py`: Shared database utility functions.
- `frontend/`: Angular-based project for the backoffice UI.
    - `src/app/models/`: TypeScript interfaces and data models.
        - `agent.ts`, `session.ts`, `chat-message.ts`, `llm-config.ts`, `llm-call-stat.ts`, `project.ts`, `user-details.ts`, `rag.ts`.
    - `src/app/services/`: Angular services for API interaction and state management.
        - `agent-service.ts`: Handles agent-related API calls (including RAG).
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
    - `tools/`: Tests for utility tools (RSS, OpenAPI parser, CSV indexer).
    - `test_persona_simulator.py`: Tests for the persona simulation logic.
- `kavalai/sql_migrations/`: SQL migration files for both `app` (agents) and `backoffice`.
    - `app/V000__agents.sql`: Initial schema for agents, sessions, runs, tasks, and chat messages.
    - `app/V001__model_call_stats.sql`: Defines `model_call_stats` for unified tracking of LLM and embedding calls.
    - `app/V002__rag.sql`: Defines `rag_index` (with `embedding` VECTOR, `collection_name`, and `source_id`).
    - `backoffice/V000__users__projects.sql`: Initial schema for users, projects, and project memberships.
    - `backoffice/V001__project_details.sql`: Adds database connection details to projects.
    - `backoffice/V002__active_project.sql`: Adds `active_project_id` to users.
- `kavalai/llm_profiles/`: Example YAML configurations for different LLM providers (OpenAI, Gemini, Anthropic, Azure, Ollama).
- `kavalai/embedding_profiles/`: Example YAML configurations for different embedding providers (OpenAI, Gemini).
- `scripts/`: Utility scripts.
    - `migrate_db.py`: Database migration tool. It supports `app` and `backoffice` migration types and uses environment variables (`KAVALAI_DB_URI`, `KAVALAI_DB_SCHEMA` for agents and `KAVALAI_BO_DB_URI`, `KAVALAI_BO_DB_SCHEMA` for backoffice) for database connections by default. It prints the masked connection URI before starting. It tracks applied migrations in `kavalai_migrations` table with checksum verification and applies them in a single transaction. Now supports `--skip-create-schema` flag to bypass schema creation while still ensuring the tracking table exists.
- `kavalai/demo_agents/`, `kavalai/demo_tasks/`, `kavalai/demo_personas/`: Sample configurations and data. Now includes `socrates.yaml` example.

## Key Technical Details
- **Backend**: Python with FastAPI, SQLAlchemy (Async), Pydantic.
- **Frontend**: Angular (using modern built-in control flow: `@if`, `@for`).
- **Database**: PostgreSQL (multiple schemas/databases).
- **Workflows**: Defined in YAML, processed by `Workflow` class in `kavalai/agents/workflow.py`. Supports REST tools with configurable URLs and basic auth via environment variables.
- **Security & Quality**:
    - Pre-commit hooks for code quality (`ruff`, `yamllint`, `pre-commit-hooks`).
    - Security scanning with `bandit` (Python) and `gitleaks` (secrets).
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
- **Notes**: Backend tests use `pytest-asyncio` for async database operations. Some tests (like `test_migrate_db.py`) use `testcontainers` for integration testing with real PostgreSQL.

## Frontend Testing (Angular)
- **Framework**: `Jasmine` + `Karma`
- **Location**: `frontend/src/app/**/*.spec.ts`
- **Command**: `cd frontend && npm test -- --watch=false`
- **Coverage Command**: `cd frontend && npm test -- --watch=false --code-coverage`
- **Notes**: Ensure all mocks are updated if service interfaces change.
