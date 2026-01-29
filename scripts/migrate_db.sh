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
python -m kavalai.migrate_db \
  --migrations kavalai/sql_migrations/backoffice \
  --host "${BACKOFFICE_DB_HOST}" \
  --port "${BACKOFFICE_DB_PORT}" \
  --user "${BACKOFFICE_DB_USER}" \
  --password "${BACKOFFICE_DB_PASSWORD}" \
  --database "${BACKOFFICE_DB_NAME}" \
  --schema "${BACKOFFICE_DB_SCHEMA}"

# App migrations
python -m kavalai.migrate_db \
  --migrations kavalai/sql_migrations/app \
  --host "${AGENTS_DB_HOST}" \
  --port "${AGENTS_DB_PORT}" \
  --user "${AGENTS_DB_USER}" \
  --password "${AGENTS_DB_PASSWORD}" \
  --database "${AGENTS_DB_NAME}" \
  --schema "${AGENTS_DB_SCHEMA}"
