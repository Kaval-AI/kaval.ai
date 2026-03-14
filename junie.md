# Junie's Project Analysis: Kaval.AI

This document serves as a guide for Junie (AI Agent) to understand the Kaval.AI project structure, components, and workflow.

## Docker Usage
Kaval.AI can be run using Docker. The provided `Dockerfile` and `entrypoint.sh` support multiple commands:
- `backoffice-migrations`: Run database migrations for the backoffice.
- `agent-migrations`: Run database migrations for the agents.
- `backoffice-server`: Start the backoffice Nginx and FastAPI server (requires `KAVALAI_BO_DB_URI` / `KAVALAI_BO_DB_SCHEMA`).
- `agent-server`: Start an agent REST server (requires `WORKFLOW_YAML_PATH` and `KAVALAI_DB_URI` / `KAVALAI_DB_SCHEMA`).
- `torproxy`: A Tor and Privoxy proxy service (using `dperson/torproxy`).

## Project Overview
Kaval.AI is an AI agent writing framework where agent steps are defined using YAML. It consists of two main parts:
- **kavalai.agents**: The core SDK/runtime that runs on client infrastructure. It handles agent logic, workflows, and its own database.
- **kavalai.backoffice**: A management UI for configuring and monitoring agents.

## Directory Structure
- `kavalai/`: Main Python package.
- `notebooks/`: Jupyter notebooks for documentation and tutorials.
- `docs/`: Sphinx documentation.
- `examples/`: Runnable example scripts for users.
- `backoffice/`: API and logic for the management UI.
- `frontend/`: Angular-based project for the backoffice UI.
- `kavalai/functionkernel.py`: Manages tool registration and execution (REST, MCP, Python), providing a unified interface for agent components. It ensures no name conflicts exist between registered tools and servers by raising a `WorkflowException` on duplicates. It can generate unified tool descriptions for LLM prompts using strictly the `protocol://[name|module].function_name` format. Supports `register_rest_tool` to register individual REST endpoints with specific HTTP methods, input/output models, and descriptions. `get_tool_descriptions()` provides clear instructions, an example `ToolCall`, and Pydantic class definitions for all registered tools to help LLMs correctly format `ToolCall` arguments. Python functions used as tools must be decorated with `@kavalai.pythontool` and explicitly registered via `register_python_tool` to be called, enhancing security. Only docstrings (parsed using `inspect.getdoc()`) are used for Python tool descriptions.
- `tests/`: Comprehensive backend test suite. `tests/test_functionkernel.py` provides 96%+ unit test coverage for `kavalai/functionkernel.py` without using broad mocks, covering REST, MCP, and Python tool registration and execution. Mock MCP servers used in tests are located in `tests/helpers/`.
- `kavalai/agents/`: SDK and agent runtime logic.
- `kavalai/llm_clients/`: Native LLM client implementations.
- `kavalai/prices/`: Pricing data for different LLM providers.
- `kavalai/tools/`: Utility tools.
- `kavalai/sql_migrations/`: SQL migration files for both `app` (agents) and `backoffice`.
- `kavalai/llm_profiles/`, `kavalai/embedding_profiles/`: Example YAML configurations.
- `kavalai/demo_agents/`, `kavalai/demo_tasks/`, `kavalai/demo_personas/`: Sample configurations and data.
        - `__init__.py`: Package initialization, defines `PACKAGE_PATH` and `MIGRATIONS_PATH`.
        - `workflow.py`: Core workflow engine (YAML to execution). Supports case-insensitive (snake_case conversion) loading of tasks and workflow models from YAML. Supports conditional task execution via the `when` field in tasks, with operators (`eq`, `not_eq`, `gt`, `gte`, `lt`, `lte`, `contains`, `is_null`, `is_not_null`, `is_true`, `len`) and logical operators (`all`, `any`, `not`). Tasks can also have a `stop` boolean field; if true, the workflow stops after executing that task. Tasks without `prompt` or `tool` are "combine" tasks, merging inputs into the specified output. REST servers support both `url` and `url_env` (with `url_env` taking precedence if defined in environment) and use `username_env` and `password_env` for basic auth. Tasks can specify an HTTP `method` for REST tool calls. Supports streaming output for both prompts (incremental) and tool/combine tasks (completion only) via an `asyncio.Queue` passed as the `queue` parameter to `run()`. Now also supports MCP (Model Context Protocol) tool calls via stdio and SSE (HTTP) clients; define servers in `mcp_servers` and reference them in tasks via `mcp_server`. MCP client sessions are initialized on the first tool call of a run and reused for subsequent calls on the same server until the run completes. The tool name is provided in `tool`, and inputs are passed as arguments to MCP tools. MCP servers support `command`, `command_env`, `url`, and `url_env`. Now supports `python_tool` for calling Python functions directly; specify the full function path (e.g., `mypackage.mymodule.myfunction`) in `python_tool`. The runtime uses `inspect` to validate that inputs match the function signature and supports both sync and async functions. Added `run_rag_task` to perform RAG-based search within workflows, supporting text resolution from context and storing results in the run data. It now also saves tool call information in the database.
        - `run_context.py`: `RunContext` model and helpers. `resolve_context_value` supports nested dotted paths (e.g., `input.criteria.keywords`). `prepare_tool_inputs` always returns a dictionary of inputs, ensuring Pydantic models are dumped to dicts. `resolve_input_info` resolves `TypeInputInfo` to values, including `load_from_history` when `agent_service` and `session_id` are set. `evaluate_condition` supports comparison operators (`eq`, `not_eq`, `gt`, `gte`, `lt`, `lte`, `contains`, `is_null`, `is_not_null`, `is_true`, `len`) and logical combinators (`all`, `any`, `not`).
        - `workflow_model.py`: Data models for workflows, including `WorkflowModel`, `Task`, `RestServer`, and `TypeInputInfo`. Includes `to_plain` utility for recursive Pydantic model serialization. Supports `temperature` parameter at both workflow and task levels (default: `None`, validated to be between 0.0 and 2.0). TypeInputInfo now supports a third type: `load_from_history` to pull values from previous runs' contexts. `Task` now includes `max_steps`, `allowed_mcp_servers`, and `timeout` for planning agent orchestration. Added `RagQueryTask` for RAG-based search within workflows.
        - `workflow_validation.py`: Dedicated module for workflow validation logic, including REST server environment variables, temperature ranges (0.0 to 2.0), and overall workflow structure. Validates `allowed_mcp_servers` in tasks.
        - `planning_agent.py`: Planning agent that orchestrates multi-step tool usage (REST + MCP) using an LLM planner. Handles tool calling, looping up to `max_steps`, and stores tool results in `planner_context`. It provides a structured `StepOutput` per iteration, which includes explanations, tool calls, and final results. The system prompt includes a complete JSON schema of `StepOutput` (including `ToolCall` and the response model) with field descriptions to guide the LLM's output. Supports optional `streamer` for real-time output and final completion streaming. Now logs tool calls and short explanations. Fixed compatibility with Gemini by ensuring at least one "user" message is sent. Improved `short_explanation` field description to encourage concise output within 50-character limit. Now gracefully handles tool execution errors by capturing the exception and passing it back to the planner in the context, allowing for self-correction. Multiple tool calls in a single step are executed in parallel using `asyncio.gather`.
        - `server.py`: REST server for agents. Supports optional HTTP basic authentication. Logs masked database connection details (using `***`), basic auth configuration status, database pooling settings, and OpenAI service tier upon startup. Includes `mask_db_uri` for secure logging of database URIs. Added `/liveness` and `/health` endpoints for K8s probes; `/health` performs a database connectivity check. `create_app_from_env_conf` supports optional parameters to override environment variables. `create_agent_app` stores the workflow in `app.state.workflow`. Supports `/run_agent` for synchronous execution and `/stream_agent` for streaming output (SSE). `stream_agent` also streams a final "output" event containing the full result (session ID and data).
        - `sessions.py`: Service for retrieving session summaries and messages. `get_sessions_summary` supports pagination and filtering by `agent_id`, `search` string (substring search in chat messages), `start_date`, and `end_date` (filter by session creation time). Added `get_session_details` to fetch full session history including grouped messages, runs with task counts, and task details.
        - `client.py`: Client for interacting with agent servers, handles schema discovery and session management. Now ensures HTTP basic auth is only used if both username and password are provided and non-empty. Added `stream_agent` method for streaming output.
        - `agent_service.py`: Service for managing agent state, sessions, runs, LLM profiles, and embedding profiles in the DB. `AgentService` is initialized with an `async_sessionmaker` to efficiently manage database connections from a pool. `get_or_create_agent` updates existing agents if description, schemas or workflow changed. Now includes `total_cost` in LLM and embedding profile views. `get_history_value` retrieves the most recent value from previous runs' context for a given key or dotted path, optimized with root-level key check in the SQL query.
        - `rag_service.py`: Service for indexing and querying text using embeddings (RAG). `RagService` is initialized with an `async_sessionmaker` or a session factory. Includes class methods `from_uri` and `from_session_maker` for flexible instantiation. `query` now returns a list of `RagServiceResult` Pydantic models and supports `keep_best` parameter to return only the best match per `source_id`. Now supports `source_ids` parameter to filter results. Added `compute_similarity_matrix` to compute similarities between multiple texts and source IDs in a single query. Added `learn_normalizer` to learn centering and normalization parameters from the index. `RagService` now accepts an optional `normalizer` instance in `__init__` which is used for all embedding computations.
    - `llm_clients/`: Native LLM client implementations.
        - `common.py`: Shared LLM client utilities, including `ModelCallStat` creation, embedding normalization, and `StreamContent` model. Now `Streamer` supports an optional `name` override for `stream_partial` and `stream_complete` to support multiple stream types (e.g., thoughts).
    - `normalizer.py`: Embedding transformation utilities including L1/L2 normalization and centering. Supports server-side centering parameter learning from RAG index (using PostgreSQL `avg()`), YAML saving/loading (from files or strings), and numpy-optimized batch operations.
        - `llm_client.py`: High-level LLM client interface. Includes `LLMClient` class that encapsulates provider selection and retries. Provides `chat_completions` for executing LLM calls with comprehensive metric collection (tokens, duration, request/response data, cost), and `compute_embeddings` for generating text embeddings. Both return a result/stats tuple, letting the caller decide whether to log stats using `_save_model_stats`. Refactored to use native OpenAI and Gemini clients instead of instructor for better structured output and stats collection. Supports configurable timeouts via `KAVALAI_LLM_TIMEOUT` (default: 30s). Added streaming support to `chat_completions`. Use `LLMClient` class directly instead of removed top-level functions. Exponential backoff retries in `with_retry` now explicitly skip 404 errors for Gemini to prevent perceived hangs on invalid models or requests, but allow retrying on 400 Bad Request which might be transient or rate-limit related in some contexts.
        - `openai_client.py`: Native OpenAI client wrapper. Supports structured outputs via `beta.chat.completions.parse` and streaming via `beta.chat.completions.stream`. Both `chat_completions` and `compute_embeddings` return a tuple of `(result, ModelCallStat)`. `chat_completions` handles images for multimodal calls and supports `stream_delta` (default: False) to stream only incremental updates. `compute_embeddings` supports normalization and custom normalizers. Supports default timeout in `__init__` and per-call overrides.
        - `gemini_client.py`: Native Gemini client wrapper. Supports structured outputs via Pydantic models passed directly to `response_schema`. Both `chat_completions` and `compute_embeddings` return a tuple of `(result, ModelCallStat)`. Now always uses streaming in `chat_completions` for both streaming and non-streaming requests, unifying text and structured output handling. Now supports `provider/model` syntax by automatically extracting the model part. Uses `GEMINI_API_KEY` from environment. Supports reasoning/thinking parameters (`reasoning_effort`, `thinking_level`, `thinking_budget`) for models that support it. Correctly handles system instructions by extracting them from messages. Now streams thoughts in real-time when `thinking_budget` is used. Supports `stream_delta` (default: False) to stream only incremental updates for both text and thoughts. Includes module-level utility functions for message conversion and config preparation. Supports default timeout in `__init__` and per-call overrides.
    - `prices/`: Pricing data for different LLM providers.
        - `common.py`: Unified Pydantic models (`ModelPricing`, `TokenPricing`) for LLM pricing and cost calculation logic.
        - `openai.py`: OpenAI pricing data using the unified models. Includes `get_openai_chat_cost` and `get_openai_embedding_cost` for calculating call costs in USD.
        - `gemini.py`: Gemini pricing data using the unified models. Includes `get_gemini_chat_cost` and `get_gemini_embedding_cost` for calculating call costs in USD.
- `examples/`: Runnable example scripts for users.
    - `business_info_agent.py`: Demonstrates `PlanningAgent` with Serper web search and scrape tools to find business information. Supports real-time streaming of agent progress to the console using the `rich` library for formatted JSON output.
    - `llm_clients/`: Examples for using the LLM client framework.
        - `01_chat_completions.py`: Streaming, multimodal (images), structured output, and reasoning examples.
        - `02_embeddings.py`: Batch embeddings, similarity search, and normalization.
        - `03_image_generation.py`: Image generation with DALL-E 3 and Imagen.
- `backoffice/`: API and logic for the management UI.
        - `__init__.py`: Package initialization, defines `PACKAGE_PATH` and `MIGRATIONS_PATH`.
        - `server.py`: FastAPI server for the backoffice API. Includes endpoints for agents, sessions, stats, projects (including membership management), and workflow visualization. `projects_rag_query` supports filtering by `source_ids` and optional `normalizer_yaml`. Now handles both backoffice and project database connection errors gracefully by returning a 503 Service Unavailable error with a descriptive message.
        - `svg.py`: Utility for generating SVG visualizations of workflows using Graphviz. Supports rendering data nodes with schema properties and resolving reference chains.
        - `db.py`: Backoffice-specific DB models (users, projects, memberships). Includes `DatabaseManager` for handling `postgresql+asyncpg` and `KAVALAI_BO_DB_URI`.
        - `project_service.py`: Service for managing project-related data and membership. Initialized with an `async_sessionmaker` to manage backoffice database connections.
    - `tools/`: Utility tools.
        - `cli_chat.py`: Command line tool for chatting with agents. Now supports Ctrl+D (EOF) to exit and uses streaming server endpoint for real-time responses.
        - `selenium_browser.py`: Selenium-based browser automation tool (FastAPI). Supports navigate, click, type, and screenshot actions.
        - `rss.py`: RSS feed parser tool.
        - `websearch/langsearch.py`: LangSearch API client for web searches. Optimized for AI applications, returns summaries and snippets in a format compatible with Bing Search API.
        - `websearch/serper.py`: Serper API client for Google Search. Supports country, language, date range and pagination.
        - `websearch/google_custom_search.py`: Google Custom Search JSON API client. Requires API key and Search Engine ID (cx).
        - `webtools/http_client.py`: Basic HTTP client for GET, POST, etc. requests with support for basic auth and proxy.
        - `webtools/serper_scraper.py`: Serper.dev Scrape API client for downloading text from websites.
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
        - `navigation-service.ts`: Manages the current page title displayed in the header.
    - `public/assets/images/`: Public assets including logos and icons.
    - `src/app/components/`: UI components organized by feature.
        - `agents-page/`, `projects-page/`, `users-page/`: CRUD interfaces for main entities.
        - `conversations-page/`, `session-detail-page/`: Agent session monitoring and debugging.
        - `configs-page/`: LLM profile and provider configuration.
        - `llm-call-stats-page/`: Detailed list of LLM calls with request/response data (paginated).
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
- `docs/`: Sphinx documentation.
    - `index.rst`: Main entry point. Updated to include `iconlogo.svg` on the landing page.
    - `conf.py`: Sphinx configuration (includes `autodoc`, `napoleon`, `viewcode`, `todo`, and `sphinx_immaterial`). Updated to use `logo.svg` as the theme logo.
    - `tutorials/`: Step-by-step guides for users and developers.
    - `architecture/`: High-level concepts and architectural overviews.
    - `api/`: Automatically generated API documentation from Python docstrings.
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
- **Project Isolation**: The backoffice manages multiple "projects". Each project can point to a different agent database. Users have an `active_project_id` stored in the database and session, which is used as the default project when navigating to the projects page. The global header includes a project selector that synchronizes the active project across all pages.

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
- `docs/tutorials/workflow_tutorial.rst`: Comprehensive guide for creating agents (Professor persona, includes examples like Socrates, Sentiment Analysis, and Jury of Judges).
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
