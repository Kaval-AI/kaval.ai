"""extract rag_index: RAG storage becomes backend-owned

RAG storage moves out of the agents migration set entirely: the Postgres RAG
backend (kavalai/rag/postgres.py) self-provisions a ``rag_collections``
registry plus one typed-vector table per collection. Existing ``rag_index``
data is NOT migrated (breaking change, accepted) — collections must be
re-indexed through the new backend.

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-08

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Dropping the table drops its indexes (including the hand-written
    # Postgres-only GIN/HNSW indexes from revision 0001) with it.
    op.drop_table("rag_index")


def downgrade() -> None:
    raise NotImplementedError(
        "rag_index is owned by the RAG backend from revision 0002 on; "
        "re-create it by downgrading to 0001 from an empty database."
    )
