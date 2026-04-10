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

import time
from typing import List, Optional, Tuple

from fastembed import TextEmbedding

from kavalai.agents.db import ModelCallStat
from kavalai.llm_clients.common import (
    create_model_call_stat,
    get_model_name,
)
from kavalai.normalizer import Normalizer, get_default_normalizer


class FastEmbedClient:
    """
    FastEmbed client for generating embeddings using ONNX Runtime.
    """

    def __init__(
        self, cache_dir: Optional[str] = None, threads: Optional[int] = None, **kwargs
    ):
        self.cache_dir = cache_dir
        self.threads = threads
        self.init_kwargs = kwargs
        self._models: dict[str, TextEmbedding] = {}

    def _get_model(self, model_name: str) -> TextEmbedding:
        if model_name not in self._models:
            self._models[model_name] = TextEmbedding(
                model_name=model_name,
                cache_dir=self.cache_dir,
                threads=self.threads,
                **self.init_kwargs,
            )
        return self._models[model_name]

    async def compute_embeddings(
        self,
        model: str,
        texts: List[str],
        normalize: bool = False,
        normalizer: Optional[Normalizer] = None,
        **kwargs,
    ) -> Tuple[List[List[float]], ModelCallStat]:
        start_time = time.perf_counter()
        model_name = get_model_name(model)

        embedding_model = self._get_model(model)

        # FastEmbed's embed returns an iterator of numpy arrays
        embeddings_iter = embedding_model.embed(texts, **kwargs)
        embeddings = [list(e) for e in embeddings_iter]

        if normalize:
            if normalizer is None:
                normalizer = get_default_normalizer()
            embeddings = normalizer.transform(embeddings)

        duration = time.perf_counter() - start_time

        stats = create_model_call_stat(
            call_type="embedding",
            model=f"fastembed/{model_name}",
            duration_sections=duration,
            batch_size=len(texts),
            total_tokens=None,  # FastEmbed doesn't easily provide token counts
            cost=0.0,
            response_data=None,
        )
        stats.currency = "USD"
        return embeddings, stats

    async def list_models(self) -> List[str]:
        return [m["model"] for m in TextEmbedding.list_supported_models()]
