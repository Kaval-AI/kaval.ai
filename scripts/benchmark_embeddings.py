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

import asyncio
import time
from typing import List

from kavalai import LLMClient
from loguru import logger

# Constants
# MODEL_NAME = "ollama/llama3.2:1b"
MODEL_NAME = "fastembed/nomic-ai/nomic-embed-text-v1.5-Q"
BENCHMARK_DURATION_SECONDS = 10
TEST_TEXT = "The quick brown fox jumps over the lazy dog. "  # ~90-100 tokens


async def run_benchmark():
    client = LLMClient(MODEL_NAME)
    logger.info(f"Starting benchmark for model: {MODEL_NAME}")
    logger.info(f"Target duration: {BENCHMARK_DURATION_SECONDS} seconds")
    start_time = time.perf_counter()
    end_time = start_time + BENCHMARK_DURATION_SECONDS

    total_embeddings = 0
    total_tokens = 0
    latencies: List[float] = []

    logger.info("Running benchmark...")

    while time.perf_counter() < end_time:
        call_start = time.perf_counter()
        try:
            # We use a single text for each call to measure per-call latency
            _, stats = await client.compute_embeddings(
                model=MODEL_NAME, texts=[TEST_TEXT] * 10
            )
            call_duration = time.perf_counter() - call_start
            latencies.append(call_duration)
            total_embeddings += 1
            total_tokens += stats.total_tokens or 0
        except Exception as e:
            logger.error(f"Error during embedding call: {e}")
            await asyncio.sleep(1)  # Brief pause on error
            if time.perf_counter() > end_time:
                break

    actual_duration = time.perf_counter() - start_time

    if not latencies:
        logger.error("Benchmark failed: No embeddings were computed.")
        return

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    throughput_embeddings = total_embeddings / actual_duration
    throughput_tokens = total_tokens / actual_duration

    logger.info("--- Benchmark Results ---")
    logger.info(f"Total time: {actual_duration:.2f} seconds")
    logger.info(f"Total embeddings: {total_embeddings}")
    logger.info(f"Total tokens: {total_tokens}")
    logger.info(f"Throughput: {throughput_embeddings:.2f} embeddings/sec")
    logger.info(f"Throughput: {throughput_tokens:.2f} tokens/sec")
    logger.info(f"Average Latency: {avg_latency * 1000:.2f} ms")
    logger.info(f"Min Latency: {min_latency * 1000:.2f} ms")
    logger.info(f"Max Latency: {max_latency * 1000:.2f} ms")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
