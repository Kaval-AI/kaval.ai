#!/usr/bin/env bash
set -e

# Usage: ./scripts/migrate_db.sh generated_env_files/dev.env
ENV_FILE="$1"

if [[ -z "$ENV_FILE" ]]; then
  echo "Usage: $0 path/to/env_file"
  exit 1
fi

# Load environment variables
export $(grep -v '^#' "$ENV_FILE" | xargs)

# Backoffice migrations
docker run --rm \
  --network kavalai_kavalai \
  -v "$(pwd)/kavalai/sql_migrations/backoffice:/flyway/sql" \
  flyway/flyway \
  -url="jdbc:postgresql://${BACKOFFICE_DB_HOST}:${BACKOFFICE_DB_PORT}/${BACKOFFICE_DB_NAME}" \
  -schemas="${BACKOFFICE_DB_SCHEMA}" \
  -user="${BACKOFFICE_DB_USER}" \
  -password="${BACKOFFICE_DB_PASSWORD}" \
  -connectRetries=1 \
  migrate

# App migrations
docker run --rm \
  --network kavalai_kavalai \
  -v "$(pwd)/kavalai/sql_migrations/app:/flyway/sql" \
  flyway/flyway \
  -url="jdbc:postgresql://${AGENTS_DB_HOST}:${AGENTS_DB_PORT}/${AGENTS_DB_NAME}" \
  -schemas="${AGENTS_DB_SCHEMA}" \
  -user="${AGENTS_DB_USER}" \
  -password="${AGENTS_DB_PASSWORD}" \
  -connectRetries=1 \
  migrate
