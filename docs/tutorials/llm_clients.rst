LLM Clients and SDK Usage
========================

The Kaval.AI SDK provides a high-level, unified interface for interacting with various Large Language Models (LLMs) and embedding providers. Our goal is to provide developers with a robust, production-ready client that abstracts away the complexities of individual provider APIs while ensuring full observability and resilience.

Key Features
------------

1. **Unified Interface:** Execute calls across different providers (OpenAI, Google Gemini, Anthropic, etc.) using a consistent API.
2. **Automatic Resilience:** Built-in retry logic with exponential backoff handles transient API errors and rate limits.
3. **Comprehensive Observability:** Every interaction is tracked via the :class:`ModelCallStat` class, capturing token usage, costs, duration, and full request/response payloads.
4. **Structured Outputs:** Native integration with Pydantic ensures that LLM responses adhere to your defined data models.
5. **Multimodal Support:** Easily incorporate images and other data types into your LLM prompts.

Core Interaction: Chat Completions
----------------------------------

The :class:`LLMClient` is the primary entry point for generating text and structured data. It manages model selection, authentication, and execution metrics.

Basic Usage
~~~~~~~~~~~

Messages are structured as a list of dictionaries, each with a **role** (system, user, or assistant) and **content**.

.. code-block:: python

   import asyncio
   from kavalai import LLMClient

   async def basic_example():
       # Initialize client with a specific model (provider/model-name)
       client = LLMClient(model="openai/gpt-4o-mini")

       messages = [
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "user", "content": "What are three interesting facts about Estonia?"}
       ]

       result, stats = await client.chat_completions(messages=messages)

       print(f"Result:\n{result}\n")
       print(f"Stats: {stats.total_tokens} tokens, duration: {stats.duration_seconds:.2f}s")

   if __name__ == "__main__":
       asyncio.run(basic_example())

**Output:**

.. code-block:: text

   Result:
   Sure! Here are three interesting facts about Estonia:

   1. **Digital Innovation**: Estonia is known for being one of the most digitally advanced countries in the world. It was the first country to offer e-residency, allowing global citizens to start and manage businesses online. The country also has a fully digital ID system that enables citizens to access a wide range of government services online.

   2. **Nature and Clean Air**: Approximately 50% of Estonia is covered by forest, making it one of the greenest countries in Europe. It is home to many national parks, nature reserves, and diverse wildlife. Estonia also boasts some of the cleanest air in the world, a testament to its commitment to environmental conservation.

   3. **Rich Cultural Heritage**: Estonia has a vibrant cultural scene, influenced by its history and various ethnic groups, including the Finno-Ugric roots from Finland. The country celebrates numerous traditional festivals, such as the Song and Dance Festival, which features thousands of singers and dancers coming together to perform. Estonia's capital, Tallinn, has a well-preserved medieval old town that is a UNESCO World Heritage site.

   Stats: 255 tokens, duration: 5.28s

Advanced Parameters
~~~~~~~~~~~~~~~~~~~

You can fine-tune the model's behavior using standard LLM parameters:

*   **temperature**: Controls deterministic vs. creative output (range: 0.0 to 2.0).
*   **max_tokens**: Sets a hard limit on the generated response length.
*   **stop**: Defines sequences that trigger the end of generation.
*   **timeout**: Overrides the default timeout for a specific call.

.. code-block:: python

   result, stats = await client.chat_completions(
       messages=messages,
       temperature=0.2,
       max_tokens=500
   )

Structured Output with Pydantic
-------------------------------

For production applications, getting structured data is crucial. Kaval.AI allows you to pass a Pydantic model as the ``response_model``, ensuring the LLM returns a validated object.

.. code-block:: python

   from pydantic import BaseModel, Field
   from kavalai import LLMClient

   class Fact(BaseModel):
       topic: str
       fact: str
       relevance_score: float

   class FactsList(BaseModel):
       facts: list[Fact]

   async def structured_example():
       client = LLMClient(model="gemini/gemini-2.0-flash")
       messages = [
           {"role": "user", "content": "Provide 3 interesting facts about chess."}
       ]

       # Passing response_model returns an instance of that model
       result, stats = await client.chat_completions(
           messages=messages,
           response_model=FactsList
       )

       for fact in result.facts:
           print(f"[{fact.topic}] ({fact.relevance_score}): {fact.fact}")

**Output:**

.. code-block:: text

   [Chess History] (0.8): The longest chess game theoretically possible is 5,949 moves.
   [Chess Strategies] (0.7): The most common opening move in chess is advancing the king's pawn two squares (e4).
   [Chess Records] (0.9): Garry Kasparov was the world's highest-rated chess player for a record 255 months overall.

Embeddings and Vector Representations
-------------------------------------

Embeddings transform text into numerical vectors, enabling semantic search and Retrieval-Augmented Generation (RAG).

.. code-block:: python

   from kavalai import LLMClient

   async def embeddings_example():
       client = LLMClient(model="openai/text-embedding-3-small")
       texts = ["Kaval.AI is an open source AI agent toolkit.",
                "D minor and F major are parallel keys."]

       embeddings, stats = await client.compute_embeddings(texts=texts, normalize=True)

       print(f"Number of embeddings: {len(embeddings)}")
       print(f"Embedding dimension: {len(embeddings[0])}\n")
       print(f"Tokens: {stats.total_tokens}")

**Output:**

.. code-block:: text

   Number of embeddings: 2
   Embedding dimension: 1536

   Tokens: 20

Real-time Streaming
-------------------

For interactive applications, Kaval.AI supports real-time streaming of responses.

.. code-block:: python

   import asyncio
   from kavalai import LLMClient, Streamer, StreamContent

   PROMPT = """
   Count from 1 to 10, say

   a. "FizzBuzz" if i is divisible by 3 and 5.
   b. "Fizz" if i is divisible by 3.
   c. "Buzz" if i is divisible by 5.
   """

   async def streaming_example():
       client = LLMClient(model="openai/gpt-4o-mini")
       messages = [{"role": "user", "content": PROMPT}]
       queue = asyncio.Queue()
       streamer = Streamer("response", queue)

       # Run the client call as a task to consume chunks from the queue
       task = asyncio.create_task(
           client.chat_completions(messages=messages, streamer=streamer, stream_delta=True)
       )

       while True:
           raw_chunk = await queue.get()
           chunk = StreamContent.model_validate_json(raw_chunk)

           if chunk.type == "partial":
               print(chunk.value, end="", flush=True)
           elif chunk.type == "complete":
               break

       _, stats = await task
       print(f"\n\nTokens: {stats.total_tokens}")

**Output:**

.. code-block:: text

   Here's the count from 1 to 10 following your rules:

   1. 1
   2. 2
   3. Fizz
   4. 4
   5. Buzz
   6. Fizz
   7. 7
   8. 8
   9. Fizz
   10. Buzz

   Let me know if you need anything else!

   Tokens: 132

Observability and Metrics
-------------------------

Every call to an LLM or embedding provider returns a :class:`ModelCallStat` object. This object is a goldmine for debugging and optimization:

*   **token_usage**: Breakdown of prompt and completion tokens.
*   **cost**: Precise cost in USD based on provider pricing (not implemented yet).
*   **duration_seconds**: Total round-trip time.
*   **request/response_data**: The exact raw payloads for full auditability.

By leveraging these metrics, you can identify slow models, monitor your budget, and refine your prompts based on actual usage data.

.. code-block:: python

   result, stats = await client.chat_completions(messages=messages)
   print(f"Call to {stats.model} took {stats.duration_seconds:.2f}s and cost ${stats.cost:.6f}")

In a production environment, these stats are automatically saved to your database, allowing you to generate reports and optimize your usage. Knowledge is power, after all!

Further Examples
----------------

You can find more advanced examples in the `notebooks/` directory.
