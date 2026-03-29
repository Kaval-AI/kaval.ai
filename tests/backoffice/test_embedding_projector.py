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

import os
import csv
import pytest
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sklearn.decomposition import IncrementalPCA
from kavalai.agents.db import RagIndex
from kavalai.backoffice.embedding_projector import download_rag_index, compute_pca


@pytest.mark.asyncio
async def test_download_rag_index(
    agents_db: AsyncSession, agents_session_maker, tmp_path
):
    # Setup: Add some items to the RAG index
    collection = "test_collection"
    items = [
        RagIndex(
            model="test_model",
            collection_name=collection,
            source_id="label1",
            content="content1",
            embedding_size=3,
            embedding=[1.0, 0.0, 0.0],
        ),
        RagIndex(
            model="test_model",
            collection_name=collection,
            source_id="label2",
            content="content2",
            embedding_size=3,
            embedding=[0.0, 1.0, 0.0],
        ),
        RagIndex(
            model="test_model",
            collection_name=collection,
            source_id="label3",
            content="content3",
            embedding_size=3,
            embedding=[0.0, 0.0, 1.0],
        ),
    ]
    agents_db.add_all(items)
    await agents_db.commit()

    csv_path = tmp_path / "rag_index.csv"

    # Execute
    await download_rag_index(agents_session_maker, collection, str(csv_path))

    # Verify
    assert os.path.exists(csv_path)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) == 3
        # Check first row
        assert rows[0][0] == "label1"
        assert [float(x) for x in rows[0][1:]] == [1.0, 0.0, 0.0]


def test_compute_pca(tmp_path):
    # Setup: Create a CSV file with embeddings
    csv_path = tmp_path / "pca_input.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Create some data that is easy to project
        # 4 points in 3D, mainly varying in X and Y
        writer.writerow(["p1", 1.0, 1.0, 0.01])
        writer.writerow(["p2", 1.1, 0.9, -0.01])
        writer.writerow(["p3", -1.0, -1.0, 0.02])
        writer.writerow(["p4", -0.9, -1.1, -0.02])

    # Execute
    ipca = compute_pca(str(csv_path), n_components=2, batch_size=2)

    # Verify
    assert isinstance(ipca, IncrementalPCA)
    assert ipca.n_components == 2

    # Check that we can transform data with it
    data = np.array(
        [[1.0, 1.0, 0.01], [1.1, 0.9, -0.01], [-1.0, -1.0, 0.02], [-0.9, -1.1, -0.02]]
    )
    transformed = ipca.transform(data)
    assert transformed.shape == (4, 2)

    # The first component should capture most of the variance (from 1,1 to -1,-1)
    # Check that it's not all zeros
    assert np.any(np.abs(transformed) > 0.1)


def test_compute_pca_empty(tmp_path):
    csv_path = tmp_path / "empty.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as _:
        pass

    with pytest.raises(ValueError, match="No data found in CSV for PCA computation."):
        compute_pca(str(csv_path))
