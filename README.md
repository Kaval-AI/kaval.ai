# Kaval.AI Enterprise Agent Management System

Kaval.AI is a Python SDK for developing AI agents.
It comes with a backoffice app that is useful for monitoring and
a/b testing different agent configurations like prompts and models.

## Local development

Apply database migrations to local and test databases.

```
./scripts/migrate_db.sh scripts/local_db.env
./scripts/migrate_db.sh scripts/test_db.env
```
