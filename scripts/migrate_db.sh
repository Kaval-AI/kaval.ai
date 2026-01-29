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
python -m kavalai.migrate_db backoffice

# App migrations
python -m kavalai.migrate_db app
