import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

import yaml
from pydantic import BaseModel
from sqlalchemy import delete
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kavalai.agents.db import EmbeddingProfile, RagIndex, Agent, db_manager
from kavalai.llm_clients.common import compute_embeddings_with_stats

logger = logging.getLogger(__name__)


class RagServiceResult(BaseModel):
    id: UUID
    embedding_profile_id: Optional[UUID] = None
    collection_name: str
    source_id: str
    content: Optional[str] = None
    embedding_size: int
    rag_metadata: dict
    similarity: float
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class RagService:
    def __init__(
        self,
        uri_or_session: str | AsyncSession,
        embedding_profile: EmbeddingProfile,
        agent: Optional[Agent] = None,
    ):
        if isinstance(uri_or_session, str):
            self.session_maker = db_manager.get_sessionmaker(uri=uri_or_session)
        else:
            # Create a context manager factory that returns this session
            from contextlib import asynccontextmanager

            @asynccontextmanager
            async def session_factory():
                yield uri_or_session

            self.session_maker = session_factory

        self.embedding_profile = embedding_profile
        self.agent = agent

    @classmethod
    def from_uri_and_path(cls, uri: str, embedding_profile_path: str):
        with open(embedding_profile_path, "r") as f:
            data = yaml.safe_load(f)
            return cls(uri, EmbeddingProfile(**data))

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

        async with self.session_maker() as session:
            embeddings = await compute_embeddings_with_stats(
                embedding_profile=self.embedding_profile,
                texts=texts,
                session=session,
                agent_id=self.agent.id if self.agent else None,
            )

            rag_items = []
            dim = len(embeddings[0])

            for i, (text, meta, emb) in enumerate(
                zip(texts, metadata_list, embeddings)
            ):
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
                session.add(rag_item)
                rag_items.append(rag_item)

            await session.commit()
            for item in rag_items:
                await session.refresh(item)

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
        async with self.session_maker() as session:
            await session.execute(stmt)
            await session.commit()

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
    ) -> list[RagServiceResult]:
        async with self.session_maker() as session:
            embeddings = await compute_embeddings_with_stats(
                embedding_profile=self.embedding_profile,
                texts=[text],
                session=session,
                agent_id=self.agent.id if self.agent else None,
            )
            query_embedding = embeddings[0]

            # Using cosine distance <=> for pgvector
            # Similarity = 1 - distance
            distance_col = RagIndex.embedding.op("<=>")(query_embedding).label(
                "distance"
            )
            stmt = (
                select(RagIndex, distance_col)
                .where(RagIndex.embedding_profile_id == self.embedding_profile.id)
                .order_by(distance_col)
                .limit(top_k)
            )

            if collection_name:
                stmt = stmt.where(RagIndex.collection_name == collection_name)

            result = await session.execute(stmt)
            rows = result.all()

            results = []
            for row in rows:
                item = row[0]
                distance = row[1]
                # Convert RagIndex object to RagServiceResult and add similarity
                res = RagServiceResult(
                    id=item.id,
                    embedding_profile_id=item.embedding_profile_id,
                    collection_name=item.collection_name,
                    source_id=item.source_id,
                    content=item.content,
                    embedding_size=item.embedding_size,
                    rag_metadata=item.rag_metadata or {},
                    similarity=1.0 - float(distance) if distance is not None else 0.0,
                    created_at=item.created_at,
                    updated_at=item.updated_at,
                )
                results.append(res)

            return results
