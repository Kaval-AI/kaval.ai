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

import numpy as np
import yaml
from typing import List, Optional, Union

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kavalai.agents.db import RagIndex


class Normalizer:
    def __init__(
        self,
        center_vector: Optional[Union[List[float], np.ndarray]] = None,
        l1: bool = False,
        l2: bool = False,
        center: bool = False,
    ):
        self.center_vector = (
            np.array(center_vector) if center_vector is not None else None
        )
        self.l1 = l1
        self.l2 = l2
        self.center_enabled = center

    def normalize_l1(self, embeddings: np.ndarray) -> np.ndarray:
        """Applies L1 normalization to a batch of embeddings."""
        norms = np.linalg.norm(embeddings, ord=1, axis=1, keepdims=True)
        return np.divide(embeddings, norms, out=np.copy(embeddings), where=norms > 0)

    def normalize_l2(self, embeddings: np.ndarray) -> np.ndarray:
        """Applies L2 normalization to a batch of embeddings."""
        norms = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
        return np.divide(embeddings, norms, out=np.copy(embeddings), where=norms > 0)

    def center(self, embeddings: np.ndarray) -> np.ndarray:
        """Subtracts the center vector from the embeddings."""
        if self.center_vector is None:
            return embeddings

        if embeddings.shape[1] != len(self.center_vector):
            raise ValueError(
                f"Embedding size {embeddings.shape[1]} does not match center vector size {len(self.center_vector)}"
            )
        return embeddings - self.center_vector

    def transform(
        self,
        embeddings: Union[List[List[float]], np.ndarray],
        l1: Optional[bool] = None,
        l2: Optional[bool] = None,
        center: Optional[bool] = None,
    ) -> List[List[float]]:
        """Applies centering and normalization in sequence."""
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
            is_single = True
        else:
            is_single = False

        l1 = l1 if l1 is not None else self.l1
        l2 = l2 if l2 is not None else self.l2
        center = center if center is not None else self.center_enabled

        result = embeddings
        if center:
            result = self.center(result)
        if l1:
            result = self.normalize_l1(result)
        if l2:
            result = self.normalize_l2(result)

        list_result = result.tolist()
        return list_result if not is_single else list_result[0]

    def save_to_yaml(self, path: str):
        """Saves the normalizer parameters to a YAML file."""
        data = {
            "l1": self.l1,
            "l2": self.l2,
            "center": self.center_enabled,
            "center_vector": self.center_vector.tolist()
            if self.center_vector is not None
            else None,
        }
        with open(path, "w") as f:
            yaml.dump(data, f)

    @classmethod
    def load_from_yaml(cls, path: str) -> "Normalizer":
        """Loads a normalizer from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(
            center_vector=data.get("center_vector"),
            l1=data.get("l1", False),
            l2=data.get("l2", False),
            center=data.get("center", False),
        )

    @classmethod
    async def learn_from_rag(
        cls,
        session: AsyncSession,
        model: str,
        collection_name: Optional[str] = None,
    ) -> "Normalizer":
        """Learns the centering vector (mean) from the RAG index."""
        stmt = select(RagIndex.embedding).where(RagIndex.model == model)
        if collection_name:
            stmt = stmt.where(RagIndex.collection_name == collection_name)

        result = await session.execute(stmt)
        embeddings = [row[0] for row in result.all() if row[0] is not None]

        if not embeddings:
            return cls()

        mean_vector = np.mean(embeddings, axis=0)
        return cls(center_vector=mean_vector)
