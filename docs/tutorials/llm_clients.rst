LLM Clients Tutorial
====================

This tutorial explains how to use the LLM clients in the Kaval.AI SDK. Our clients are lightweight wrappers around official libraries (like OpenAI and Google GenAI), designed to unify the interface for common Generative AI operations.

Why use Kaval.AI clients?
-------------------------

If you are new to Python or AI development, you might wonder why we don't just use the official libraries directly.

1. **Unified Interface:** Whether you use OpenAI or Gemini, the code remains almost identical.
2. **Built-in Resilience:** We include automatic retry logic with exponential backoff for common API errors.
3. **Integrated Metrics:** Every call automatically tracks token usage, costs, and execution time.
4. **Structured Output:** Easy integration with Pydantic for getting guaranteed data formats.

Basic Chat Completion
---------------------

The most common way to interact with an LLM is through chat completions. In Kaval.AI, we use a list of "messages" to represent the conversation.

Each message has a **role** (system, user, or assistant) and **content**.

*   **system**: Sets the behavior of the AI (e.g., "You are a chess Grandmaster").
*   **user**: Your prompt or question.
*   **assistant**: The AI's previous responses (used for conversation history).

Let's look at a "nerdy" example: analyzing a chess position.

.. code-block:: python

   import asyncio
   from kavalai.llm_clients.llm_client import LLMClient

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

Writing a Good Prompt
~~~~~~~~~~~~~~~~~~~~~

A good prompt is clear and provides context. Instead of "Fix my code", try "You are a Python expert. Identify the bug in this function and explain why it occurs."

Multimodal Execution (Using Images)
-----------------------------------

Modern AI models can "see". This is called "multimodal" execution. You can send images along with your text. This is useful for identifying computer parts or analyzing diagrams.

.. code-block:: python

   import asyncio
   from kavalai.llm_clients.llm_client import LLMClient

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

**Expected Output:**

.. code-block:: text

   Identification: This is an NVIDIA GeForce RTX 4090 graphics card. It features a triple-fan cooling system and requires a 16-pin power connector...

Streaming and Delta Mode
------------------------

When you use ChatGPT, the text appears word-by-word. This is called **streaming**.

Kaval.AI supports streaming via the `Streamer` class. You can also enable `stream_delta`, which only sends the *new* characters (the "delta") rather than the whole message every time.

**Trade-offs:**
*   **On (Streaming):** Feels much faster for users because they see progress immediately.
*   **Off (Non-streaming):** Simpler to write code for, but the user has to wait for the entire response to finish before seeing anything.

.. code-block:: python

   import asyncio
   from kavalai.llm_clients.llm_client import LLMClient
   from kavalai.llm_clients.common import Streamer, StreamContent

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

**Expected Output:**

.. code-block:: text

   AI is writing: Silicon minds begin to wake,
   Through the data, paths they take.
   Learning patterns, fast and deep,
   Secrets that the bitstreams keep.

Structured Output with Pydantic
-------------------------------

If you need the AI to return data in a specific format (like a JSON object or a Python class), use Pydantic. This is extremely powerful for building apps.

.. code-block:: python

   from pydantic import BaseModel
   from kavalai.llm_clients.llm_client import LLMClient

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

**Expected Output:**

.. code-block:: text

   CPU: Ryzen 9
   RAM: 32GB

Further Examples
----------------

You can find more advanced examples in the `examples/llm_clients/` directory:

*   `01_chat_completions.py`: Detailed usage of chat, streaming, and multimodal.
*   `02_embeddings.py`: How to turn text into numbers (vectors) for search and AI memory.
