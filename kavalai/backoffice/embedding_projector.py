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
from typing import Callable, AsyncContextManager
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sklearn.decomposition import IncrementalPCA
import numpy as np

from kavalai.agents.db import RagIndex


async def download_rag_index(
    session_maker: Callable[[], AsyncContextManager[AsyncSession]],
    collection_name: str,
    output_csv_path: str,
):
    """
    Downloads rag index to specified CSV file using a streaming cursor.
    The first column is label (source_id), the next columns represent the embeddings.

    Args:
        session_maker: A callable that returns an async session context manager.
        collection_name: The name of the collection to download.
        output_csv_path: The path where the CSV file will be saved.
    """
    async with session_maker() as session:
        stmt = (
            select(RagIndex)
            .where(RagIndex.collection_name == collection_name)
            .execution_options(yield_per=100)
        )
        result = await session.stream_scalars(stmt)

        with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            async for item in result:
                if item.embedding:
                    row = [item.source_id] + list(item.embedding)
                    writer.writerow(row)


def compute_pca(
    csv_path: str, n_components: int = 2, batch_size: int = 100
) -> IncrementalPCA:
    """
    Given the CSV file, computes PCA model from the dataset using scikit-learn's IncrementalPCA.
    Returns the fitted IncrementalPCA model.

    Args:
        csv_path: Path to the CSV file containing labels and embeddings.
        n_components: Number of principal components to compute.
        batch_size: The number of rows to process at once for incremental training.

    Returns:
        IncrementalPCA: The fitted PCA model.
    """
    # Fit IncrementalPCA model
    ipca = IncrementalPCA(n_components=n_components)
    batch = []
    has_data = False

    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row:
                continue
            has_data = True
            batch.append([float(x) for x in row[1:]])
            if len(batch) >= batch_size:
                ipca.partial_fit(np.array(batch))
                batch = []

        # Final partial_fit for remaining rows (if any)
        if batch and len(batch) >= n_components:
            ipca.partial_fit(np.array(batch))
        elif batch:
            pass

    if not has_data:
        raise ValueError("No data found in CSV for PCA computation.")

    return ipca
