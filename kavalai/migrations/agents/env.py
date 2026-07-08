"""Alembic environment for the *agents* migration set.

Covers the agent-runtime tables (``agents``, ``sessions``, ``runs``) and the
built-in SQL history backend tables (``chat_messages``, ``tasks``,
``model_call_stats``), plus — until the RAG backend becomes fully
self-provisioning — ``rag_index``.
"""

from kavalai.db import Base
from kavalai.migrations.common import (
    agents_include_object,
    agents_render_item,
    run_migrations,
)

run_migrations(
    Base.metadata,
    render_item=agents_render_item,
    include_object=agents_include_object,
)
