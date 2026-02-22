import asyncio

import numpy as np

from kavalai import LLMClient, Normalizer


async def basic_embeddings_example():
    """
    Demonstrates how to generate embeddings for a list of strings.
    Embeddings are vector representations of text that capture semantic meaning.
    """
    print("\n--- Basic Embeddings ---")
    client = LLMClient(model="openai/text-embedding-3-small")
    texts = [
        "Kaval.AI is an AI agent framework.",
        "I love building intelligent agents.",
    ]

    # compute_embeddings returns a tuple of (embeddings, stats)
    # embeddings is a list of lists of floats.
    embeddings, stats = await client.compute_embeddings(texts=texts)

    print(f"Generated {len(embeddings)} embeddings.")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"Stats: {stats.total_tokens} tokens, Cost: ${stats.cost:.6f}")


async def batch_usage_example():
    """
    Demonstrates processing a larger batch of texts.
    Batching is more efficient and often required for indexing large datasets.
    """
    print("\n--- Batch Embeddings ---")
    client = LLMClient(model="openai/text-embedding-3-small")

    # Example of many small texts
    texts = [f"This is document number {i}" for i in range(10)]

    embeddings, stats = await client.compute_embeddings(texts=texts)
    print(f"Processed {len(embeddings)} texts in one call.")
    print(f"Total tokens used: {stats.total_tokens}")


async def similarity_and_distance_example():
    """
    Demonstrates how to calculate the similarity between two texts using their embeddings.
    Cosine similarity is commonly used, which is equivalent to the dot product of L2-normalized vectors.
    """
    print("\n--- Similarity and Distance ---")
    client = LLMClient(model="openai/text-embedding-3-small")

    texts = [
        "The cat sat on the mat.",  # Text A
        "A feline rested on the rug.",  # Text B (Similar to A)
        "The weather is sunny today.",  # Text C (Different)
    ]

    # We use normalize=True to get L2-normalized embeddings directly
    embeddings, _ = await client.compute_embeddings(texts=texts, normalize=True)

    # Convert to numpy for easy math
    vec_a = np.array(embeddings[0])
    vec_b = np.array(embeddings[1])
    vec_c = np.array(embeddings[2])

    # Cosine Similarity = Dot Product (since they are normalized)
    sim_ab = np.dot(vec_a, vec_b)
    sim_ac = np.dot(vec_a, vec_c)

    print(f"Similarity (A, B): {sim_ab:.4f} (Should be high)")
    print(f"Similarity (A, C): {sim_ac:.4f} (Should be lower)")

    # Euclidean Distance
    dist_ab = np.linalg.norm(vec_a - vec_b)
    print(f"Euclidean Distance (A, B): {dist_ab:.4f}")


async def normalizer_example():
    """
    Demonstrates using the Normalizer class for advanced preprocessing.
    Centering and L2 normalization can improve retrieval performance in RAG systems.
    """
    print("\n--- Advanced Normalization ---")

    # Create a normalizer that performs L2 normalization
    normalizer = Normalizer(l2=True)

    client = LLMClient(model="openai/text-embedding-3-small")
    texts = ["Custom normalized embedding."]

    # The client will use the normalizer's transform() method on the raw embeddings
    embeddings, _ = await client.compute_embeddings(texts=texts, normalizer=normalizer)

    # Verify it's normalized (norm should be 1.0)
    norm = np.linalg.norm(embeddings[0])
    print(f"Embedding norm with L2 normalizer: {norm:.4f}")


async def main():
    await basic_embeddings_example()
    await batch_usage_example()
    await similarity_and_distance_example()
    await normalizer_example()


if __name__ == "__main__":
    asyncio.run(main())
