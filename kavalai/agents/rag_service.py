from kavalai.agents.db import EmbeddingProfile, RagIndex
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from kavalai.llm_clients.common import compute_embeddings


class RagService:
    def __init__(
        self,
        db_session: AsyncSession,
        embedding_profile: EmbeddingProfile,
    ):
        self.db = db_session
        self.embedding_profile = embedding_profile

    async def batch_index(
        self,
        texts: list[str],
        metadata_list: list[dict],
        collection_name: str = "default",
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

        for text, meta, emb in zip(texts, metadata_list, embeddings):
            item_data = {
                "embedding_profile_id": self.embedding_profile.id,
                "collection_name": collection_name,
                "content": text,
                "embedding_size": dim,
                "embedding": emb,
                "rag_metadata": meta,
            }
            rag_item = RagIndex(**item_data)
            self.db.add(rag_item)
            rag_items.append(rag_item)

        await self.db.commit()
        for item in rag_items:
            await self.db.refresh(item)

        return rag_items

    async def index(
        self,
        text: str,
        source_metadata: Optional[dict] = None,
        collection_name: str = "default",
    ):
        """Index a single text blob with the metadata."""
        return (
            await self.batch_index([text], [source_metadata or {}], collection_name)
        )[0]

    async def query(
        self,
        text: str,
        top_k: int = 5,
        collection_name: Optional[str] = None,
    ) -> list[RagIndex]:
        embeddings = await compute_embeddings(
            llm_profile=self.embedding_profile, texts=[text]
        )
        query_embedding = embeddings[0]

        # Using cosine distance <=> for pgvector
        stmt = (
            select(RagIndex)
            .where(RagIndex.embedding_profile_id == self.embedding_profile.id)
            .order_by(RagIndex.embedding.op("<=>")(query_embedding))
            .limit(top_k)
        )

        if collection_name:
            stmt = stmt.where(RagIndex.collection_name == collection_name)

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def batch_query(
        self,
        texts: list[str],
        top_k: int = 5,
        collection_name: Optional[str] = None,
    ) -> list[list[RagIndex]]:
        results = []
        for text in texts:
            results.append(
                await self.query(text, top_k=top_k, collection_name=collection_name)
            )
        return results
