from kavalai.agents.db import EmbeddingProfile, RagIndex
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from kavalai.llm_clients.common import compute_embeddings


class RagService:
    def __init__(self, db_session: AsyncSession, embedding_profile: EmbeddingProfile):
        self.db = db_session
        self.embedding_profile = embedding_profile

    async def batch_index(
        self, texts: list[str], metadata_list: list[dict]
    ) -> list[RagIndex]:
        if not texts:
            return []

        if len(texts) != len(metadata_list):
            raise ValueError(
                "The number of texts and metadata dictionaries must be the same."
            )

        embeddings = await compute_embeddings(
            llm_profile=self.embedding_profile, texts=texts
        )

        rag_items = []
        dim = len(embeddings[0])
        embedding_field = f"embedding_{dim}"

        if not hasattr(RagIndex, embedding_field):
            raise ValueError(
                f"Unsupported embedding dimension: {dim}. RagIndex does not have a column for it."
            )

        for text, meta, emb in zip(texts, metadata_list, embeddings):
            item_data = {
                "embedding_profile_id": self.embedding_profile.id,
                embedding_field: emb,
                "mime_type": "text/plain",
                "text_content": text,
                "metadata_": meta,
            }
            rag_item = RagIndex(**item_data)
            self.db.add(rag_item)
            rag_items.append(rag_item)

        await self.db.commit()
        for item in rag_items:
            await self.db.refresh(item)

        return rag_items

    async def index(self, text: str, metadata: Optional[dict] = None):
        """Index a single text blob with the metadata."""
        return (await self.batch_index([text], [metadata or {}]))[0]

    async def query(self, text: str, top_k: int = 5) -> list[RagIndex]:
        embeddings = await compute_embeddings(
            llm_profile=self.embedding_profile, texts=[text]
        )
        query_embedding = embeddings[0]
        dim = len(query_embedding)
        embedding_field_name = f"embedding_{dim}"

        if not hasattr(RagIndex, embedding_field_name):
            raise ValueError(f"Unsupported embedding dimension: {dim}.")

        embedding_col = getattr(RagIndex, embedding_field_name)

        # Using cosine distance <=> for pgvector
        # We need to use func.public.cosine_distance or similar if we want to be explicit,
        # but usually pgvector supports operators.
        # However, in SQLAlchemy we can use op('<=>')

        stmt = (
            select(RagIndex)
            .where(RagIndex.embedding_profile_id == self.embedding_profile.id)
            .order_by(embedding_col.op("<=>")(query_embedding))
            .limit(top_k)
        )

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def batch_query(
        self, texts: list[str], top_k: int = 5
    ) -> list[list[RagIndex]]:
        results = []
        for text in texts:
            results.append(await self.query(text, top_k=top_k))
        return results
