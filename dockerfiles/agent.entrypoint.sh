#!/bin/bash
set -e

# Function to run agent migrations
run_agent_migrations() {
    echo "Running agent migrations..."
    python -m kavalai.migrate_db \
        --migrations kavalai/sql_migrations/app \
        --host "$AGENTS_DB_HOST" \
        --port "${AGENTS_DB_PORT:-5432}" \
        --user "$AGENTS_DB_USER" \
        --password "$AGENTS_DB_PASSWORD" \
        --database "$AGENTS_DB_NAME" \
        --schema "${AGENTS_DB_SCHEMA:-public}"
}

# Function to run agent server
run_agent_server() {
    echo "Starting agent server..."
    if [ -z "$WORKFLOW_YAML_PATH" ]; then
        echo "Error: WORKFLOW_YAML_PATH environment variable is required to run agent server."
        exit 1
    fi
    # Use environment variables for DB connection in kavalai.agents.server
    exec python -m kavalai.agents.server "$WORKFLOW_YAML_PATH" --port "${AGENT_PORT:-8001}" --host "${AGENT_HOST:-0.0.0.0}"
}

case "$1" in
    agent-migrations)
        run_agent_migrations
        ;;
    agent-server)
        run_agent_server
        ;;
    *)
        echo "Usage: $0 {agent-migrations|agent-server}"
        exit 1
        ;;
esac
