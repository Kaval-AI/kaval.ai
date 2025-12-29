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

# Run Flyway container
docker run --rm \
  --network kavalai_kavalai \
  -v "$(pwd)/sql_migrations:/flyway/sql" \
  flyway/flyway \
  -url="jdbc:postgresql://${POSTGRES_DB_HOST}:${POSTGRES_DB_PORT}/${POSTGRES_DB_NAME}" \
  -schemas="${POSTGRES_DB_SCHEMA}" \
  -user="${POSTGRES_DB_USER}" \
  -password="${POSTGRES_DB_PASSWORD}" \
  -connectRetries=1 \
  migrate
