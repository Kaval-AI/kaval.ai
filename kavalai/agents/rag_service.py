import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import delete, select, func
from sqlalchemy.ext.asyncio import AsyncSession

from kavalai.agents.db import RagIndex, Agent, db_manager
from kavalai.llm_clients.common import compute_embeddings

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

            # Using cosine distance <=> for pgvector
            # Similarity = 1 - distance
            distance_col = RagIndex.embedding.op("<=>")(query_embedding).label(
                "distance"
            )
            stmt = select(RagIndex, distance_col).where(RagIndex.model == self.model)

            if collection_name:
                stmt = stmt.where(RagIndex.collection_name == collection_name)

            if source_ids:
                stmt = stmt.where(RagIndex.source_id.in_(source_ids))

            if keep_best:
                # Define a subquery to find the minimum distance for each source_id
                sub_stmt = (
                    select(
                        RagIndex.source_id,
                        func.min(distance_col).label("min_distance"),
                    )
                    .where(RagIndex.model == self.model)
                    .group_by(RagIndex.source_id)
                )

                if collection_name:
                    sub_stmt = sub_stmt.where(
                        RagIndex.collection_name == collection_name
                    )

                if source_ids:
                    sub_stmt = sub_stmt.where(RagIndex.source_id.in_(source_ids))

                sub_stmt = sub_stmt.subquery()

                # Join the main query with the subquery to keep only the best results per source_id
                stmt = stmt.join(
                    sub_stmt,
                    (RagIndex.source_id == sub_stmt.c.source_id)
                    & (distance_col == sub_stmt.c.min_distance),
                )

            stmt = stmt.order_by(distance_col).limit(top_k)

            result = await session.execute(stmt)
            rows = result.all()

            results = []
            for row in rows:
                item = row[0]
                distance = row[1]
                # Convert RagIndex object to RagServiceResult and add similarity
                res = RagServiceResult(
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
                results.append(res)

            return results

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

            agg_func = func.min if method == "min" else func.avg

            # Construct a single query with one column per text embedding
            cols = [RagIndex.source_id]
            for i, emb in enumerate(embeddings):
                distance_col = RagIndex.embedding.op("<=>")(emb)
                cols.append(agg_func(distance_col).label(f"dist_{i}"))

            stmt = (
                select(*cols)
                .where(
                    RagIndex.model == self.model,
                    RagIndex.source_id.in_(source_ids),
                )
                .group_by(RagIndex.source_id)
            )

            result = await session.execute(stmt)
            rows = result.all()

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
