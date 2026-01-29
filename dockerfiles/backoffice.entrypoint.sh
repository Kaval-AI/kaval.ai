#!/bin/bash
set -e

# Function to run backoffice migrations
run_backoffice_migrations() {
    echo "Running backoffice migrations..."
    python -m kavalai.migrate_db \
        --migrations kavalai/sql_migrations/backoffice \
        --host "$BACKOFFICE_DB_HOST" \
        --port "${BACKOFFICE_DB_PORT:-5432}" \
        --user "$BACKOFFICE_DB_USER" \
        --password "$BACKOFFICE_DB_PASSWORD" \
        --database "$BACKOFFICE_DB_NAME" \
        --schema "${BACKOFFICE_DB_SCHEMA:-public}"
}

# Function to run backoffice server
run_backoffice_server() {
    echo "Starting Nginx..."
    service nginx start

    echo "Starting backoffice server..."
    # FRONTEND_URL should be set in environment if it's not the default
    exec python -m kavalai.backoffice.server
}

case "$1" in
    backoffice-migrations)
        run_backoffice_migrations
        ;;
    backoffice-server)
        run_backoffice_server
        ;;
    *)
        echo "Usage: $0 {backoffice-migrations|backoffice-server}"
        exit 1
        ;;
esac
