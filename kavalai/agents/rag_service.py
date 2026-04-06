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

import os
from datetime import datetime
from typing import Optional, Union, Callable, AsyncContextManager
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import delete, select, func, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import aliased

from kavalai.agents.db import RagIndex, Agent, db_manager
from kavalai.llm_clients.llm_client import LLMClient
from kavalai.normalizer import Normalizer


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
        query_index (Optional[int]): Index of the query in batch queries (for batch_query results).
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
    query_index: Optional[int] = None


class RagService:
    """
    Service for indexing and querying text using embeddings (Retrieval-Augmented Generation).

    This service provides methods to batch index text, delete items, query similarities,
    and compute similarity matrices against indexed content.
    """

    def __init__(
        self,
        session_maker: Union[
            async_sessionmaker[AsyncSession],
            Callable[[], AsyncContextManager[AsyncSession]],
        ],
        model: str,
        agent: Optional[Agent] = None,
        normalizer: Optional[Normalizer] = None,
    ):
        """
        Initialize the RagService.

        Args:
            session_maker (Union[async_sessionmaker[AsyncSession], Callable[[], AsyncContextManager[AsyncSession]]]):
                Async session maker or a factory that returns an async context manager for the session.
            model (str): The name of the embedding model to use (e.g., "openai/text-embedding-3-small").
            agent (Optional[Agent]): Optional Agent object to associate with this service.
            normalizer (Optional[Normalizer]): Optional normalizer to use for embeddings.
        """
        self.session_maker = session_maker
        self.model = model
        self.agent = agent
        self.normalizer = normalizer
        self.llm_client = LLMClient(model)

    @classmethod
    def from_uri(
        cls,
        uri: str,
        model: str,
        agent: Optional[Agent] = None,
        normalizer: Optional[Normalizer] = None,
    ) -> "RagService":
        """
        Create a RagService from a database URI.

        Args:
            uri (str): Database URI.
            model (str): The name of the embedding model to use.
            agent (Optional[Agent]): Optional Agent object to associate with this service.
            normalizer (Optional[Normalizer]): Optional normalizer to use for embeddings.

        Returns:
            RagService: A new instance of RagService.
        """
        session_maker = db_manager.get_sessionmaker(uri=uri)
        return cls(session_maker, model, agent, normalizer)

    @classmethod
    def from_session_maker(
        cls,
        session_maker: async_sessionmaker[AsyncSession],
        model: str,
        agent: Optional[Agent] = None,
        normalizer: Optional[Normalizer] = None,
    ) -> "RagService":
        """
        Create a RagService from a session maker.

        Args:
            session_maker (async_sessionmaker[AsyncSession]): Async session maker for the database.
            model (str): The name of the embedding model to use.
            agent (Optional[Agent]): Optional Agent object to associate with this service.
            normalizer (Optional[Normalizer]): Optional normalizer to use for embeddings.

        Returns:
            RagService: A new instance of RagService.
        """
        return cls(session_maker, model, agent, normalizer)

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
            embeddings, stats = await self.llm_client.compute_embeddings(
                texts=texts,
                normalizer=self.normalizer,
            )
            session.add(stats)

            rag_items = []
            dim = len(embeddings[0])

            for i, (content, meta, emb) in enumerate(
                zip(texts, metadata_list, embeddings)
            ):
                item_data = {
                    "model": self.model,
                    "collection_name": collection_name,
                    "source_id": source_ids[i] if source_ids else "default",
                    "content": content,
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
            embeddings, stats = await self.llm_client.compute_embeddings(
                texts=[text],
                normalizer=self.normalizer,
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

    async def batch_query(
        self,
        texts: list[str],
        top_k: int = 5,
        collection_name: Optional[str] = None,
        source_ids: Optional[list[str]] = None,
    ) -> list[list[RagServiceResult]]:
        """
        Query the indexed items for similarities to multiple input texts in a single database call.

        This method uses PostgreSQL CROSS JOIN LATERAL to efficiently process multiple queries
        in a single round trip to the database, significantly improving performance for batch operations.

        Args:
            texts (list[str]): List of query texts to search for.
            top_k (int): Number of top results to return per query. Defaults to 5.
            collection_name (Optional[str]): If provided, filter by collection name.
            source_ids (Optional[list[str]]): If provided, filter by source identifiers.

        Returns:
            list[list[RagServiceResult]]: A list of result lists, where each inner list contains
                                          the top_k results for the corresponding query text.
        """
        if not texts:
            return []

        async with self.session_maker() as session:
            # Get schema from environment variable if possible, similar to other parts of the system
            schema = os.getenv("KAVALAI_DB_SCHEMA")
            if schema:
                await session.execute(text(f"SET search_path TO {schema}, public"))

            # Compute embeddings for all query texts
            embeddings, stats = await self.llm_client.compute_embeddings(
                texts=texts,
                normalizer=self.normalizer,
            )
            session.add(stats)

            # Build the CROSS JOIN LATERAL query
            query = self._build_batch_query_sql(
                embeddings=embeddings,
                top_k=top_k,
                collection_name=collection_name,
                source_ids=source_ids,
            )

            # Execute the query
            result = await session.execute(query)
            rows = result.all()

            # Group results by query_index
            results_by_query: dict[int, list[RagServiceResult]] = {
                i: [] for i in range(len(texts))
            }

            for row in rows:
                rag_result = RagServiceResult(
                    id=row.id,
                    model=row.model,
                    collection_name=row.collection_name,
                    source_id=row.source_id,
                    content=row.content,
                    embedding_size=row.embedding_size,
                    rag_metadata=row.metadata or {},
                    similarity=1.0 - float(row.distance)
                    if row.distance is not None
                    else 0.0,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                    query_index=int(row.query_idx)
                    - 1,  # Convert 1-indexed to 0-indexed
                )
                results_by_query[int(row.query_idx) - 1].append(rag_result)

            # Return results in order
            return [results_by_query[i] for i in range(len(texts))]

    async def batch_query_with_join(
        self,
        texts: list[str],
        top_k: int = 5,
        collection_name: Optional[str] = None,
        join_table: Optional[str] = None,
        join_condition: Optional[str] = None,
        join_columns: Optional[list[str]] = None,
        additional_where: Optional[str] = None,
        keep_best: bool = False,
    ) -> list[list[dict]]:
        """
        Query indexed items and join with another table in a single SQL query using CTE.

        This is useful when you need to filter RAG results by attributes in another table
        (e.g., product category, price, availability) without passing large lists of IDs.

        Args:
            texts (list[str]): List of query texts to search for.
            top_k (int): Number of top results to return per query. Defaults to 5.
            collection_name (Optional[str]): If provided, filter by collection name.
            join_table (Optional[str]): Table to join with (e.g., "products p").
            join_condition (Optional[str]): Join condition (e.g., "p.id::text = r.source_id").
            join_columns (Optional[list[str]]): Additional columns to select from joined table.
            additional_where (Optional[str]): Additional WHERE clause for filtering joined table.
            keep_best (bool): If True, only the best result per source_id is returned for each query.

        Returns:
            list[list[dict]]: A list of result lists, where each inner list contains
                             dictionaries with RAG results + joined table columns.

        Example:
            results = await service.batch_query_with_join(
                texts=["wireless headphones", "laptop stand"],
                top_k=10,
                collection_name="products",
                join_table="products p",
                join_condition="p.id::text = r.source_id",
                join_columns=["p.name", "p.price", "p.category", "p.in_stock"],
                additional_where="p.category = 'electronics' AND p.in_stock = true"
            )
        """
        if not texts:
            return []

        async with self.session_maker() as session:
            # Set schema if configured
            schema = os.getenv("KAVALAI_DB_SCHEMA")
            if schema:
                await session.execute(text(f"SET search_path TO {schema}, public"))

            # Compute embeddings
            embeddings, stats = await self.llm_client.compute_embeddings(
                texts=texts,
                normalizer=self.normalizer,
            )
            session.add(stats)

            # Build source filter for CTE if additional_where is provided
            source_filter = None
            if join_table and additional_where:
                # Extract table alias from join_table (e.g., "products p" -> "p")
                table_parts = join_table.split()
                _ = table_parts[-1] if len(table_parts) > 1 else table_parts[0]
                source_filter = f"""
                    EXISTS (
                        SELECT 1 FROM {join_table}
                        WHERE {join_condition.replace('r.source_id', 'rag_index.source_id')}
                        AND {additional_where}
                    )
                """

            # Build CTE
            cte_sql, params = self.build_batch_query_cte(
                embeddings=embeddings,
                top_k=top_k,
                collection_name=collection_name,
                source_filter_sql=source_filter,
                keep_best=keep_best,
            )

            # Build final query
            select_columns = [
                "r.id",
                "r.source_id",
                "r.content",
                "r.similarity",
                "r.query_idx",
            ]

            if join_columns:
                select_columns.extend(join_columns)

            select_clause = ", ".join(select_columns)

            if join_table and join_condition:
                query_sql = f"""
                    WITH {cte_sql}
                    SELECT {select_clause}
                    FROM rag_results r
                    JOIN {join_table} ON {join_condition}
                    ORDER BY r.query_idx, r.similarity DESC
                """
            else:
                query_sql = f"""
                    WITH {cte_sql}
                    SELECT {select_clause}
                    FROM rag_results r
                    ORDER BY r.query_idx, r.similarity DESC
                """

            result = await session.execute(text(query_sql).bindparams(**params))
            rows = result.all()

            # Group results by query_index
            results_by_query: dict[int, list[dict]] = {i: [] for i in range(len(texts))}

            for row in rows:
                row_dict = dict(row._mapping)
                query_idx = row_dict.pop("query_idx") - 1  # Convert to 0-indexed
                results_by_query[query_idx].append(row_dict)

            return [results_by_query[i] for i in range(len(texts))]

    def build_batch_query_cte(
        self,
        embeddings: list[list[float]],
        top_k: int,
        collection_name: Optional[str] = None,
        source_filter_sql: Optional[str] = None,
        keep_best: bool = False,
    ) -> tuple[str, dict]:
        """
        Build a CTE (Common Table Expression) for batch vector search that can be embedded in larger queries.

        This allows you to join RAG results with other tables in a single query.

        Args:
            embeddings (list[list[float]]): Pre-computed embeddings for the queries.
            top_k (int): Number of results per query.
            collection_name (Optional[str]): Filter by collection name.
            source_filter_sql (Optional[str]): Additional SQL WHERE clause to filter sources.
                                               Can reference other CTEs or tables.
                                               Example: "EXISTS (SELECT 1 FROM filtered_hotels fh WHERE fh.id = rag_index.source_id)"
            keep_best (bool): If True, only the best result per source_id is returned for each query.
                             Useful when a single source is split into multiple indexed items.

        Returns:
            tuple[str, dict]: (CTE SQL without WITH keyword, parameter dict)

        Example:
            embeddings, _ = await llm_client.compute_embeddings(texts)
            cte_sql, params = service.build_batch_query_cte(embeddings, top_k=10)

            query = text(f'''
                WITH {cte_sql},
                hotel_data AS (
                    SELECT h.*, r.similarity, r.query_idx
                    FROM rag_results r
                    JOIN hotels h ON h.id = r.source_id::uuid
                )
                SELECT * FROM hotel_data ORDER BY query_idx, similarity DESC
            ''').bindparams(**params)
        """
        params = self._build_cte_params(embeddings, collection_name)
        vector_array = self._build_vector_array(len(embeddings))
        where_clause = self._build_cte_where_clause(collection_name, source_filter_sql)

        if keep_best:
            # Use DISTINCT ON to get only the best result per source_id for each query
            # Apply LIMIT inside LATERAL to control how many RAG entries we scan per keyword
            # Then DISTINCT ON deduplicates to get one per (query_idx, source_id)
            cte_sql = f"""rag_results AS (
                SELECT DISTINCT ON (v.query_idx, results.source_id)
                    results.id,
                    results.model,
                    results.collection_name,
                    results.source_id,
                    results.content,
                    results.embedding_size,
                    results.metadata,
                    results.created_at,
                    results.updated_at,
                    results.distance,
                    (1.0 - results.distance) as similarity,
                    v.query_idx
                FROM
                    unnest({vector_array}) WITH ORDINALITY AS v(query_vector, query_idx)
                CROSS JOIN LATERAL (
                    SELECT
                        id,
                        model,
                        collection_name,
                        source_id,
                        content,
                        embedding_size,
                        metadata,
                        created_at,
                        updated_at,
                        (embedding <=> v.query_vector) as distance
                    FROM
                        rag_index
                    WHERE
                        {where_clause}
                    ORDER BY
                        (embedding <=> v.query_vector) ASC
                    LIMIT :top_k
                ) AS results
                ORDER BY v.query_idx ASC, results.source_id, results.distance ASC
            )"""
        else:
            cte_sql = f"""rag_results AS (
                SELECT
                    results.id,
                    results.model,
                    results.collection_name,
                    results.source_id,
                    results.content,
                    results.embedding_size,
                    results.metadata,
                    results.created_at,
                    results.updated_at,
                    results.distance,
                    (1.0 - results.distance) as similarity,
                    v.query_idx
                FROM
                    unnest({vector_array}) WITH ORDINALITY AS v(query_vector, query_idx)
                CROSS JOIN LATERAL (
                    SELECT
                        id,
                        model,
                        collection_name,
                        source_id,
                        content,
                        embedding_size,
                        metadata,
                        created_at,
                        updated_at,
                        (embedding <=> v.query_vector) as distance
                    FROM
                        rag_index
                    WHERE
                        {where_clause}
                    ORDER BY
                        (embedding <=> v.query_vector) ASC
                    LIMIT :top_k
                ) AS results
                ORDER BY v.query_idx ASC, results.distance ASC
            )"""

        params["top_k"] = top_k
        return cte_sql, params

    def _build_cte_params(
        self, embeddings: list[list[float]], collection_name: Optional[str]
    ) -> dict:
        """Build parameter dictionary for CTE."""
        params = {"model": self.model}
        for i, embedding in enumerate(embeddings):
            params[f"vector_{i}"] = str(embedding)
        if collection_name:
            params["collection_name"] = collection_name
        return params

    def _build_vector_array(self, num_embeddings: int) -> str:
        """Build the ARRAY[...] expression for vector embeddings."""
        vector_parts = [f"CAST(:vector_{i} AS vector)" for i in range(num_embeddings)]
        return f"ARRAY[{', '.join(vector_parts)}]"

    def _build_cte_where_clause(
        self, collection_name: Optional[str], source_filter_sql: Optional[str]
    ) -> str:
        """Build WHERE clause for CTE."""
        where_clauses = ["model = :model"]
        if collection_name:
            where_clauses.append("collection_name = :collection_name")
        if source_filter_sql:
            where_clauses.append(f"({source_filter_sql})")
        return " AND ".join(where_clauses)

    def _build_batch_query_sql(
        self,
        embeddings: list[list[float]],
        top_k: int,
        collection_name: Optional[str] = None,
        source_ids: Optional[list[str]] = None,
    ):
        """
        Build a CROSS JOIN LATERAL SQL query for batch vector search.

        This uses the technique described in:
        https://www.murhabazi.com/batch-vector-search-pgvector-postgresql-cross-lateral-joins
        """
        # Build the array of vectors for UNNEST
        vector_array_parts = []
        params = {}

        for i, embedding in enumerate(embeddings):
            param_name = f"vector_{i}"
            params[param_name] = str(embedding)
            vector_array_parts.append(f"CAST(:{param_name} AS vector)")

        vector_array = f"ARRAY[{', '.join(vector_array_parts)}]"

        # Build WHERE clause filters
        where_clauses = ["model = :model"]
        params["model"] = self.model

        if collection_name:
            where_clauses.append("collection_name = :collection_name")
            params["collection_name"] = collection_name

        if source_ids:
            where_clauses.append("source_id = ANY(:source_ids)")
            params["source_ids"] = source_ids

        where_clause = " AND ".join(where_clauses)

        # Build the complete query
        query_sql = f"""
        SELECT
            results.*,
            v.query_idx
        FROM
            unnest({vector_array}) WITH ORDINALITY AS v(query_vector, query_idx)
        CROSS JOIN LATERAL (
            SELECT
                id,
                model,
                collection_name,
                source_id,
                content,
                embedding_size,
                metadata,
                created_at,
                updated_at,
                (embedding <=> v.query_vector) as distance
            FROM
                rag_index
            WHERE
                {where_clause}
            ORDER BY
                (embedding <=> v.query_vector) ASC
            LIMIT :top_k
        ) AS results
        ORDER BY v.query_idx ASC, results.distance ASC
        """

        params["top_k"] = top_k

        return text(query_sql).bindparams(**params)

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
            embeddings, stats = await self.llm_client.compute_embeddings(
                texts=texts,
                normalizer=self.normalizer,
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
