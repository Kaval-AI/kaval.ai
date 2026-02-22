LLM Clients Tutorial
====================

Welcome, class! Today we are going to explore the wonderful world of Large Language Models (LLMs) and how to interact with them using the Kaval.AI SDK. Our clients are designed to be lightweight, elegant wrappers around official libraries (like OpenAI and Google GenAI), providing you with a unified and serene interface for all your Generative AI operations.

Why use Kaval.AI clients?
-------------------------

If you are new to Python or AI development, you might wonder why we don't just use the official libraries directly. Think of Kaval.AI as a helpful teaching assistant that handles the complex details for you:

1. **Unified Interface:** Whether you use OpenAI or Gemini, the code remains almost identical. Consistency is the key to mastery!
2. **Built-in Resilience:** We include automatic retry logic with exponential backoff for common API errors. Even the best models sometimes have a "bad day."
3. **Integrated Metrics:** Every call automatically tracks token usage, costs, and execution time using the :class:`ModelCallStat` class.
4. **Structured Output:** Easy integration with Pydantic for getting guaranteed data formats.

Basic Chat Completion
---------------------

The most common way to interact with an LLM is through chat completions. In Kaval.AI, we use a list of "messages" to represent the conversation.

Each message has a **role** (system, user, or assistant) and **content**.

*   **system**: Sets the behavior of the AI (e.g., "You are a chess Grandmaster").
*   **user**: Your prompt or question.
*   **assistant**: The AI's previous responses (used for conversation history).

Let's look at an example: analyzing a chess position.

.. code-block:: python

   import asyncio
   from kavalai import LLMClient

   async def analyze_chess():
       # Initialize the client for OpenAI's GPT-4o
       client = LLMClient(model="openai/gpt-4o")

       messages = [
           {"role": "system", "content": "You are a chess expert. Use algebraic notation."},
           {"role": "user", "content": "What is the best move for white in the Ruy Lopez opening after 3... a6?"}
       ]

       # chat_completions returns (result, stats)
       result, stats = await client.chat_completions(messages=messages)

       print(f"AI Analysis: {result}")
       print(f"Tokens Used: {stats.total_tokens}")
       print(f"Cost: ${stats.cost:.4f}")

   if __name__ == "__main__":
       asyncio.run(analyze_chess())

**Expected Output:**

.. code-block:: text

   AI Analysis: The most common and strongest move for White is 4. Ba4, maintaining the pressure on the knight on c6. Another option is 4. Bxc6, the Exchange Variation...
   Tokens Used: 85
   Cost: $0.0004

Fine-Tuning the Response
~~~~~~~~~~~~~~~~~~~~~~~~

When calling `chat_completions`, you can pass additional parameters to guide the AI's "creativity" and behavior. These are passed as keyword arguments:

*   **temperature**: Controls the "randomness" or "creativity" of the output. A low temperature (e.g., 0.1) makes the model more deterministic and focused, while a high temperature (e.g., 0.8) makes it more diverse and creative.
*   **max_tokens**: Limits the length of the response.
*   **stop**: A list of strings that, if encountered, will cause the model to stop generating further tokens.

.. code-block:: python

   # A focused, deterministic response
   result, stats = await client.chat_completions(
       messages=messages,
       temperature=0.1
   )

Embeddings: Turning Words into Wisdom
-------------------------------------

Now, let's talk about **Embeddings**. Imagine if every sentence could be represented as a point in a vast, multi-dimensional space. Sentences with similar meanings would be close together, while unrelated ones would be far apart. This "vector" representation is what we call an embedding.

In Kaval.AI, we use the :class:`LLMClient` to compute these embeddings. They are fundamental for building Knowledge Bases and the :class:`RagIndex` (Retrieval-Augmented Generation).

.. code-block:: python

   from kavalai import LLMClient

   async def learn_embeddings():
       client = LLMClient(model="openai/text-embedding-3-small")
       texts = ["Kaval.AI makes agent development easy.", "I love learning about AI."]

       # compute_embeddings returns (list of vectors, stats)
       embeddings, stats = await client.compute_embeddings(texts=texts)

       print(f"Generated {len(embeddings)} embeddings.")
       print(f"First 5 dimensions of first embedding: {embeddings[0][:5]}")

Embeddings allow your agents to search through thousands of documents to find exactly the right information to answer a user's question. We call this "Retrieval," and it's what makes RAG so powerful!

Multimodal Execution (Using Images)
-----------------------------------

Modern AI models can "see". This is called "multimodal" execution. You can send images along with your text. This is useful for identifying computer parts or analyzing diagrams.

.. code-block:: python

   import asyncio
   from kavalai import LLMClient

   async def identify_hardware(image_base64: str):
       client = LLMClient(model="openai/gpt-4o")

       messages = [
           {
               "role": "user",
               "content": [
                   {"type": "input_text", "text": "What computer component is this?"},
                   {
                       "type": "input_image",
                       "image_url": f"data:image/jpeg;base64,{image_base64}"
                   }
               ]
           }
       ]

       result, _ = await client.chat_completions(messages=messages)
       print(f"Identification: {result}")

Streaming and Delta Mode
------------------------

When you use ChatGPT, the text appears word-by-word. This is called **streaming**.

Kaval.AI supports streaming via the :class:`Streamer` class. You can also enable `stream_delta`, which only sends the *new* characters (the "delta") rather than the whole message every time.

**Trade-offs:**
*   **On (Streaming):** Feels much faster for users because they see progress immediately.
*   **Off (Non-streaming):** Simpler to write code for, but the user has to wait for the entire response to finish before seeing anything.

.. code-block:: python

   import asyncio
   from kavalai import LLMClient, Streamer, StreamContent

   async def stream_poem():
       client = LLMClient(model="openai/gpt-4o")
       messages = [{"role": "user", "content": "Write a 4-line poem about artificial intelligence."}]

       queue = asyncio.Queue()
       streamer = Streamer("poem_stream", queue)

       # Start the request in the background
       task = asyncio.create_task(
           client.chat_completions(messages=messages, streamer=streamer, stream_delta=True)
       )

       print("AI is writing: ", end="", flush=True)
       while True:
           # Get a chunk of data from the queue
           raw_chunk = await queue.get()
           chunk = StreamContent.model_validate_json(raw_chunk)

           if chunk.type == "partial":
               print(chunk.value, end="", flush=True)
           elif chunk.type == "complete":
               break

       await task

Structured Output with Pydantic
-------------------------------

If you need the AI to return data in a specific format (like a JSON object or a Python class), use Pydantic. This is extremely powerful for building apps.

.. code-block:: python

   from pydantic import BaseModel
   from kavalai import LLMClient

   class ComputerSpec(BaseModel):
       cpu: str
       ram_gb: int
       is_gaming_pc: bool

   async def get_specs():
       client = LLMClient(model="openai/gpt-4o")
       messages = [{"role": "user", "content": "Extract specs from: 'I have a Ryzen 9 with 32GB RAM for gaming.'"}]

       result, _ = await client.chat_completions(
           messages=messages,
           response_model=ComputerSpec
       )

       print(f"CPU: {result.cpu}")
       print(f"RAM: {result.ram_gb}GB")

Monitoring and Statistics (ModelCallStat)
-----------------------------------------

Every time you call an LLM, Kaval.AI tracks the details of that interaction in a :class:`ModelCallStat` object. This is essential for monitoring your application's performance and costs.

The `stats` object returned by `chat_completions` or `compute_embeddings` contains:

*   **total_tokens**: The total number of tokens used (prompt + completion).
*   **cost**: The calculated cost of the call in USD.
*   **duration_seconds**: How long the API call took.
*   **model**: The full name of the model used.
*   **request_data**: The exact payload sent to the provider.
*   **response_data**: The raw response received from the provider.

.. code-block:: python

   result, stats = await client.chat_completions(messages=messages)
   print(f"Call to {stats.model} took {stats.duration_seconds:.2f}s and cost ${stats.cost:.6f}")

In a production environment, these stats are automatically saved to your database, allowing you to generate reports and optimize your usage. Knowledge is power, after all!

Further Examples
----------------

You can find more advanced examples in the `examples/llm_clients/` directory:

*   `01_chat_completions.py`: Detailed usage of chat, streaming, multimodal, and reasoning models.
*   `02_embeddings.py`: How to turn text into numbers (vectors) for search and AI memory.
