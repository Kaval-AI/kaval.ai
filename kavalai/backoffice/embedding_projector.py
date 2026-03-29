"""
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
you may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import csv
import pickle
import base64
import json
import tempfile
import os
from typing import Callable, AsyncContextManager, Optional
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sklearn.decomposition import IncrementalPCA
import numpy as np
from loguru import logger

from kavalai.agents.db import RagIndex
from kavalai.backoffice.db import Project, ProjectCache
from kavalai.llm_clients.common import Streamer


async def download_rag_index(
    session_maker: Callable[[], AsyncContextManager[AsyncSession]],
    collection_name: str,
    output_csv_path: str,
    streamer: Optional[Streamer] = None,
):
    """
    Downloads rag index to specified CSV file using a streaming cursor.
    The first column is label (source_id), the next columns represent the embeddings.

    Args:
        session_maker: A callable that returns an async session context manager.
        collection_name: The name of the collection to download.
        output_csv_path: The path where the CSV file will be saved.
        streamer: Optional streamer for progress reporting.
    """
    async with session_maker() as session:
        # Get total count
        count_stmt = (
            select(func.count())
            .select_from(RagIndex)
            .where(RagIndex.collection_name == collection_name)
        )
        total_count = await session.scalar(count_stmt) or 0

        stmt = (
            select(RagIndex)
            .where(RagIndex.collection_name == collection_name)
            .execution_options(yield_per=100)
        )
        result = await session.stream_scalars(stmt)

        with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            count = 0
            async for item in result:
                if item.embedding:
                    # Use content as label, fallback to source_id if content is empty
                    label = item.content or item.source_id
                    row = [label] + list(item.embedding)
                    writer.writerow(row)
                    count += 1
                    if count % 100 == 0:
                        msg = f"Downloaded {count}/{total_count} items..."
                        if streamer:
                            await streamer.stream_partial(msg)

            msg = f"Finished downloading {count}/{total_count} items."
            logger.info(msg)
            if streamer:
                await streamer.stream_partial(msg)


async def compute_pca(
    csv_path: str,
    n_components: int = 2,
    batch_size: int = 100,
    streamer: Optional[Streamer] = None,
) -> IncrementalPCA:
    """
    Given the CSV file, computes PCA model from the dataset using scikit-learn's IncrementalPCA.
    Returns the fitted IncrementalPCA model.

    Args:
        csv_path: Path to the CSV file containing labels and embeddings.
        n_components: Number of principal components to compute.
        batch_size: The number of rows to process at once for incremental training.
        streamer: Optional streamer for progress reporting.

    Returns:
        IncrementalPCA: The fitted PCA model.
    """
    # Fit IncrementalPCA model
    ipca = IncrementalPCA(n_components=n_components)
    batch = []
    has_data = False
    row_count = 0
    batch_count = 0

    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row:
                continue
            has_data = True
            batch.append([float(x) for x in row[1:]])
            row_count += 1
            if len(batch) >= batch_size:
                ipca.partial_fit(np.array(batch))
                batch = []
                batch_count += 1
                if batch_count % 5 == 0:
                    msg = f"Processed {row_count} rows for PCA..."
                    if streamer:
                        await streamer.stream_partial(msg)

        # Final partial_fit for remaining rows (if any)
        if batch and len(batch) >= n_components:
            ipca.partial_fit(np.array(batch))
            msg = f"Processed final {row_count} rows for PCA."
            logger.info(msg)
            if streamer:
                await streamer.stream_partial(msg)
        elif batch:
            pass

    if not has_data:
        raise ValueError("No data found in CSV for PCA computation.")

    return ipca


async def train_pca(
    bo_session_maker: Callable[[], AsyncContextManager[AsyncSession]],
    agents_session_maker: Callable[[], AsyncContextManager[AsyncSession]],
    project_name: str,
    collection_name: str,
    streamer: Optional[Streamer] = None,
):
    """
    Trains PCA model for a given collection and stores it in the project cache.

    Args:
        bo_session_maker: Callable returning an async backoffice session.
        agents_session_maker: Callable returning an async agents session.
        project_name: Name of the project.
        collection_name: Name of the collection in RagIndex.
        streamer: Optional streamer for progress reporting.
    """

    if streamer:
        await streamer.stream_partial(
            f"Starting PCA training for collection: {collection_name}"
        )

    # 1. Find project_id by project_name
    async with bo_session_maker() as bo_session:
        stmt = select(Project).where(Project.name == project_name)
        result = await bo_session.execute(stmt)
        project = result.scalar_one_or_none()
        if not project:
            raise ValueError(f"Project '{project_name}' not found.")
        project_id = project.id

    # 2. Download embeddings to a temporary CSV
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_csv_path = tmp.name

    try:
        if streamer:
            await streamer.stream_partial("Downloading embeddings...")
        await download_rag_index(
            agents_session_maker,
            collection_name,
            output_csv_path=tmp_csv_path,
            streamer=streamer,
        )

        # 3. Fit PCA model
        if streamer:
            await streamer.stream_partial("Computing PCA model...")
        ipca = await compute_pca(
            tmp_csv_path,
            n_components=2,
            batch_size=100,
            streamer=streamer,
        )

        # 4. Transform a sample of 500 points
        if streamer:
            await streamer.stream_partial("Generating sample points...")
        sample_points = []
        with open(tmp_csv_path, "r", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            count = 0
            batch = []
            labels = []
            for row in reader:
                if not row:
                    continue
                labels.append(row[0])
                batch.append([float(x) for x in row[1:]])
                count += 1
                if count >= 500:
                    break

            if batch:
                transformed = ipca.transform(np.array(batch))
                for i in range(len(transformed)):
                    sample_points.append(
                        {
                            "label": labels[i],
                            "x": transformed[i][0],
                            "y": transformed[i][1],
                        }
                    )

        # 5. Store both the model and example points in project_cache table
        if streamer:
            await streamer.stream_partial("Storing results in cache...")
        model_data = base64.b64encode(pickle.dumps(ipca)).decode("utf-8")
        sample_data = json.dumps(sample_points)

        async with bo_session_maker() as bo_session:
            # Upsert model
            model_cache_name = f"pca_model_{collection_name}"
            stmt = select(ProjectCache).where(
                ProjectCache.project_id == project_id,
                ProjectCache.name == model_cache_name,
            )
            res = await bo_session.execute(stmt)
            cache_entry = res.scalar_one_or_none()
            if cache_entry:
                cache_entry.value = model_data
            else:
                bo_session.add(
                    ProjectCache(
                        project_id=project_id, name=model_cache_name, value=model_data
                    )
                )

            # Upsert sample points
            sample_cache_name = f"pca_sample_train_data_{collection_name}"
            stmt = select(ProjectCache).where(
                ProjectCache.project_id == project_id,
                ProjectCache.name == sample_cache_name,
            )
            res = await bo_session.execute(stmt)
            cache_entry = res.scalar_one_or_none()
            if cache_entry:
                cache_entry.value = sample_data
            else:
                bo_session.add(
                    ProjectCache(
                        project_id=project_id, name=sample_cache_name, value=sample_data
                    )
                )
            await bo_session.commit()

        if streamer:
            await streamer.stream_complete("PCA training completed successfully.")
        logger.info(f"PCA training completed for collection {collection_name}")

    finally:
        if os.path.exists(tmp_csv_path):
            os.remove(tmp_csv_path)
