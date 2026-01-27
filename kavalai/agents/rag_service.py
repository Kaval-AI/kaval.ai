from typing import Optional

from sqlalchemy import delete
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kavalai.agents.db import EmbeddingProfile, RagIndex
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
        source_ids: Optional[list[str]] = None,
    ) -> list[RagIndex]:
        if not texts:
            return []

        if len(texts) != len(metadata_list):
            raise ValueError(
                "The number of texts and metadata dictionaries must be the same."
            )

        if source_ids and len(texts) != len(source_ids):
            raise ValueError("The number of texts and source_ids must be the same.")

        embeddings = await compute_embeddings(
            llm_profile=self.embedding_profile, texts=texts
        )

        rag_items = []
        dim = len(embeddings[0])

        for i, (text, meta, emb) in enumerate(zip(texts, metadata_list, embeddings)):
            item_data = {
                "embedding_profile_id": self.embedding_profile.id,
                "collection_name": collection_name,
                "source_id": source_ids[i] if source_ids else "default",
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

    async def delete_by_source_ids(
        self,
        collection_name: str,
        source_ids: list[str],
    ):
        """Delete items from a collection by their source_ids."""
        stmt = delete(RagIndex).where(
            RagIndex.collection_name == collection_name,
            RagIndex.source_id.in_(source_ids),
        )
        await self.db.execute(stmt)
        await self.db.commit()

    async def index(
        self,
        text: str,
        source_metadata: Optional[dict] = None,
        collection_name: str = "default",
        source_id: str = "default",
    ):
        """Index a single text blob with the metadata."""
        return (
            await self.batch_index(
                [text], [source_metadata or {}], collection_name, source_ids=[source_id]
            )
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
