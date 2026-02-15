"""
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import delete, select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from kavalai.agents.db import RagIndex, Agent, db_manager
from kavalai.llm_clients.llm_client import compute_embeddings
from kavalai.normalizer import Normalizer

logger = logging.getLogger(__name__)


class RagServiceResult(BaseModel):
    """
    Represents a single result from a RAG query.

    Attributes:
        id (UUID): Unique identifier of the indexed item.
        model (str): The embedding model used for this item.
        collection_name (str): The name of the collection this item belongs to.
        source_id (str): An external identifier for the source of this item.
        content (Optional[str]): The original text content that was indexed.
        embedding_size (int): The dimension of the embedding vector.
        rag_metadata (dict): Additional metadata associated with the item.
        similarity (float): The similarity score (1.0 - distance) relative to the query.
        created_at (Optional[datetime]): Timestamp when the item was created.
        updated_at (Optional[datetime]): Timestamp when the item was last updated.
    """

    id: UUID
    model: str
    collection_name: str
    source_id: str
    content: Optional[str] = None
    embedding_size: int
    rag_metadata: dict
    similarity: float
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class RagService:
    """
    Service for indexing and querying text using embeddings (Retrieval-Augmented Generation).

    This service provides methods to batch index text, delete items, query similarities,
    and compute similarity matrices against indexed content.
    """

    def __init__(
        self,
        uri_or_session: str | AsyncSession,
        model: str,
        agent: Optional[Agent] = None,
    ):
        """
        Initialize the RagService.

        Args:
            uri_or_session (str | AsyncSession): Database URI or an active AsyncSession.
            model (str): The name of the embedding model to use (e.g., "openai/text-embedding-3-small").
            agent (Optional[Agent]): Optional Agent object to associate with this service.
        """
        if isinstance(uri_or_session, str):
            self.session_maker = db_manager.get_sessionmaker(uri=uri_or_session)
        else:
            # Create a context manager factory that returns this session
            from contextlib import asynccontextmanager

            @asynccontextmanager
            async def session_factory():
                yield uri_or_session

            self.session_maker = session_factory
        self.model = model
        self.agent = agent

    async def batch_index(
        self,
        *,
        texts: list[str],
        metadata_list: list[dict],
        source_ids: Optional[list[str]] = None,
        collection_name: str = "default",
    ) -> list[RagIndex]:
        """
        Index multiple text items in a single batch.

        Args:
            texts (list[str]): List of text strings to index.
            metadata_list (list[dict]): List of metadata dictionaries for each text.
            source_ids (Optional[list[str]]): Optional list of source identifiers.
                                              If not provided, "default" is used.
            collection_name (str): Name of the collection to add items to. Defaults to "default".

        Returns:
            list[RagIndex]: List of created RagIndex database objects.

        Raises:
            ValueError: If the lengths of texts, metadata_list, or source_ids do not match.
        """
        if not texts:
            return []

        if len(texts) != len(metadata_list):
            raise ValueError(
                "The number of texts and metadata dictionaries must be the same."
            )

        if source_ids and len(texts) != len(source_ids):
            raise ValueError("The number of texts and source_ids must be the same.")

        async with self.session_maker() as session:
            embeddings, stats = await compute_embeddings(
                model=self.model,
                texts=texts,
            )
            session.add(stats)

            rag_items = []
            dim = len(embeddings[0])

            for i, (text, meta, emb) in enumerate(
                zip(texts, metadata_list, embeddings)
            ):
                item_data = {
                    "model": self.model,
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
        """
        Delete items from a collection by their source identifiers.

        Args:
            collection_name (str): The name of the collection.
            source_ids (list[str]): List of source identifiers to delete.
        """
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
        """
        Index a single text blob with metadata.

        Args:
            text (str): The text content to index.
            source_metadata (Optional[dict]): Metadata to associate with the text.
            collection_name (str): Name of the collection. Defaults to "default".
            source_id (str): Source identifier. Defaults to "default".

        Returns:
            RagIndex: The created RagIndex database object.
        """
        return (
            await self.batch_index(
                texts=[text],
                metadata_list=[source_metadata or {}],
                collection_name=collection_name,
                source_ids=[source_id],
            )
        )[0]

    async def query(
        self,
        text: str,
        top_k: int = 5,
        collection_name: Optional[str] = None,
        source_ids: Optional[list[str]] = None,
        keep_best: bool = False,
    ) -> list[RagServiceResult]:
        """
        Query the indexed items for similarities to the input text.

        Args:
            text (str): The query text.
            top_k (int): Number of top results to return. Defaults to 5.
            collection_name (Optional[str]): If provided, filter by collection name.
            source_ids (Optional[list[str]]): If provided, filter by source identifiers.
            keep_best (bool): If True, only the best result per source_id is returned.
                             Useful when a single source is split into multiple indexed items.

        Returns:
            list[RagServiceResult]: List of results with similarity scores.
        """
        async with self.session_maker() as session:
            embeddings, stats = await compute_embeddings(
                model=self.model,
                texts=[text],
            )
            session.add(stats)
            query_embedding = embeddings[0]

            stmt = self._build_query_stmt(
                query_embedding=query_embedding,
                top_k=top_k,
                collection_name=collection_name,
                source_ids=source_ids,
                keep_best=keep_best,
            )

            result = await session.execute(stmt)
            rows = result.all()

            return [self._map_row_to_result(row[0], row[1]) for row in rows]

    def _build_query_stmt(
        self,
        query_embedding: list[float],
        top_k: int,
        collection_name: Optional[str] = None,
        source_ids: Optional[list[str]] = None,
        keep_best: bool = False,
    ):
        """Build the SQLAlchemy statement for the RAG query."""
        distance_col = RagIndex.embedding.op("<=>")(query_embedding).label("distance")

        if keep_best:
            # Use DISTINCT ON to get one row per source_id.
            # We order by source_id, then distance, then id as a tie-breaker.
            distinct_stmt = (
                select(RagIndex, distance_col.label("distance"))
                .distinct(RagIndex.source_id)
                .where(RagIndex.model == self.model)
            )

            if collection_name:
                distinct_stmt = distinct_stmt.where(
                    RagIndex.collection_name == collection_name
                )

            if source_ids:
                distinct_stmt = distinct_stmt.where(RagIndex.source_id.in_(source_ids))

            distinct_stmt = distinct_stmt.order_by(
                RagIndex.source_id, distance_col, RagIndex.id
            )

            sub = distinct_stmt.subquery()
            # Select RagIndex columns from the subquery
            stmt = select(aliased(RagIndex, sub), sub.c.distance).order_by(
                sub.c.distance
            )
        else:
            stmt = select(RagIndex, distance_col).where(RagIndex.model == self.model)

            if collection_name:
                stmt = stmt.where(RagIndex.collection_name == collection_name)

            if source_ids:
                stmt = stmt.where(RagIndex.source_id.in_(source_ids))

            stmt = stmt.order_by(distance_col)

        return stmt.limit(top_k)

    def _map_row_to_result(self, item: RagIndex, distance: float) -> RagServiceResult:
        """Map a database row to a RagServiceResult."""
        return RagServiceResult(
            id=item.id,
            model=item.model,
            collection_name=item.collection_name,
            source_id=item.source_id,
            content=item.content,
            embedding_size=item.embedding_size,
            rag_metadata=item.rag_metadata or {},
            similarity=1.0 - float(distance) if distance is not None else 0.0,
            created_at=item.created_at,
            updated_at=item.updated_at,
        )

    async def compute_similarity_matrix(
        self,
        texts: list[str],
        source_ids: list[str],
        method: str = "min",
    ) -> list[list[float]]:
        """
        Compute a similarity matrix between multiple texts and multiple source identifiers.

        This method generates embeddings for all input texts and performs a single database
        query to find similarities against all specified source_ids.

        Args:
            texts (list[str]): List of query texts (rows in the matrix).
            source_ids (list[str]): List of source identifiers to compare against (columns in the matrix).
            method (str): Aggregate method to use when multiple items exist for a source_id.
                          "min" (default) uses the shortest distance (highest similarity).
                          "avg" uses the average distance.

        Returns:
            list[list[float]]: A 2D matrix where matrix[i][j] is the similarity between
                               texts[i] and source_ids[j].
        """
        if not texts or not source_ids:
            return [[0.0 for _ in source_ids] for _ in texts]

        async with self.session_maker() as session:
            embeddings, stats = await compute_embeddings(
                model=self.model,
                texts=texts,
            )
            session.add(stats)

            stmt = self._build_similarity_matrix_stmt(embeddings, source_ids, method)
            result = await session.execute(stmt)
            rows = result.all()

            return self._process_similarity_matrix_rows(rows, texts, source_ids)

    def _build_similarity_matrix_stmt(
        self,
        embeddings: list[list[float]],
        source_ids: list[str],
        method: str,
    ):
        """Build the SQLAlchemy statement for similarity matrix computation."""
        agg_func = func.min if method == "min" else func.avg

        # Construct a single query with one column per text embedding
        cols = [RagIndex.source_id]
        for i, emb in enumerate(embeddings):
            distance_col = RagIndex.embedding.op("<=>")(emb)
            cols.append(agg_func(distance_col).label(f"dist_{i}"))

        return (
            select(*cols)
            .where(
                RagIndex.model == self.model,
                RagIndex.source_id.in_(source_ids),
            )
            .group_by(RagIndex.source_id)
        )

    def _process_similarity_matrix_rows(
        self,
        rows: list,
        texts: list[str],
        source_ids: list[str],
    ) -> list[list[float]]:
        """Process database rows into a similarity matrix."""
        # Map source_id to index for quick lookup
        source_id_to_idx = {sid: i for i, sid in enumerate(source_ids)}
        # Initialize matrix with 0.0
        matrix = [[0.0 for _ in range(len(source_ids))] for _ in range(len(texts))]

        for row in rows:
            sid = row.source_id
            if sid in source_id_to_idx:
                s_idx = source_id_to_idx[sid]
                for t_idx in range(len(texts)):
                    # Retrieve distance from the dynamically named column
                    dist = getattr(row, f"dist_{t_idx}")
                    # Similarity = 1 - Distance
                    matrix[t_idx][s_idx] = (
                        1.0 - float(dist) if dist is not None else 0.0
                    )

        return matrix

    async def learn_normalizer(
        self, collection_name: Optional[str] = None
    ) -> Normalizer:
        """Learns a normalizer from the current RAG index."""
        async with self.session_maker() as session:
            return await Normalizer.learn_from_rag(
                session, model=self.model, collection_name=collection_name
            )
