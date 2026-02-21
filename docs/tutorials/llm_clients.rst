LLM Clients Tutorial
===================

This tutorial covers the usage of the LLM clients in Kaval.AI SDK.
While Kaval.AI LLM clients are designed to be used internally with the SDK, they are still useful
abstractions for common LLM tasks.

Kaval.AI currently only supports models provided by OpenAI and Gemini.

Comparison with Other Frameworks
--------------------------------

The Kaval.AI LLM client framework shares similarities with other industry-standard frameworks like `instructor`, but it is tailored for the Kaval.AI agent ecosystem.

**Similarities:**

* **Pydantic Integration:** Like `instructor`, we use Pydantic models to define the expected structure of LLM responses, ensuring type safety and easy data validation.
* **Multi-Provider Support:** We provide a unified interface for different LLM providers (currently OpenAI and Gemini).
* **Retry Logic:** Built-in support for retrying failed API calls with exponential backoff.

**Differences:**

* **Agent-Centric Design:** Kaval.AI clients are optimized for use within autonomous agents, with built-in support for tracking metrics like token usage, costs, and execution duration across complex workflows.
* **Native Provider Features:** We leverage native provider features (like OpenAI's Structured Outputs and Gemini's `response_schema`) directly, rather than relying solely on prompting techniques, which improves reliability and reduces latency.
* **Streaming & Multimodal:** First-class support for streaming responses and multimodal inputs (images) across all supported providers.
* **Integrated Pricing:** Automatic cost calculation based on the latest pricing models for each provider.

Examples
--------

All examples in this tutorial can be found in the `examples/llm_clients/` directory of the repository.

* `GitHub: Chat Completions <https://github.com/kavalai/kaval.ai/blob/main/examples/llm_clients/01_chat_completions.py>`_
* `GitHub: Embeddings <https://github.com/kavalai/kaval.ai/blob/main/examples/llm_clients/02_embeddings.py>`_

Simple Usage
------------

The recommended way to use LLM clients is through the `LLMClient` class in the `kavalai.llm_clients.llm_client` module.

Basic Prompt with Pydantic Response
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Kaval.AI uses Pydantic models to ensure structured responses from LLMs.

.. code-block:: python

   import asyncio
   from pydantic import BaseModel
   from kavalai.llm_clients.llm_client import LLMClient

   class Greeting(BaseModel):
       message: str
       language: str

   async def main():
       messages = [
           {"role": "user", "content": "Say hello in French"}
       ]

       # Use OpenAI
       client_oa = LLMClient(model="openai/gpt-4o")
       result, stats = await client_oa.chat_completions(
           messages=messages,
           response_model=Greeting
       )
       print(f"OpenAI: {result.message} ({result.language})")

       # Use Gemini
       client_gem = LLMClient(model="gemini/gemini-2.0-flash")
       result, stats = await client_gem.chat_completions(
           messages=messages,
           response_model=Greeting
       )
       print(f"Gemini: {result.message} ({result.language})")

   if __name__ == "__main__":
       asyncio.run(main())

**Demo Output:**

.. code-block:: text

   OpenAI: Bonjour! (French)
   Gemini: Salut! (French)

Computing Embeddings
--------------------

You can compute embeddings for a list of strings using the `LLMClient.compute_embeddings` method.

.. code-block:: python

   from kavalai.llm_clients.llm_client import LLMClient

   async def get_embeddings():
       texts = ["Kaval.AI is an agent management system", "LLMs are powerful"]

       client = LLMClient(model="openai/text-embedding-3-small")
       embeddings, stats = await client.compute_embeddings(
           texts=texts
       )
       print(f"Computed {len(embeddings)} embeddings")
       return embeddings

**Demo Output:**

.. code-block:: text

   Computing embeddings for 2 strings...
   Computed 2 embeddings
   Text: 'Kaval.AI is an agent management system' -> Vector size: 1536
   Text: 'LLMs are powerful' -> Vector size: 1536
   Stats: Tokens: 12, Cost: $0.000001

Using Images (Multimodal)
-------------------------

Both OpenAI and Gemini clients support multimodal inputs. You can provide images using a standard OpenAI-compatible format in the `messages` list.

.. code-block:: python

   from kavalai.llm_clients.llm_client import LLMClient
   from pydantic import BaseModel

   class ImageDescription(BaseModel):
       description: str
       objects_found: list[str]

   async def describe_image(image_base64: str):
       messages = [
           {
               "role": "user",
               "content": [
                   {"type": "text", "text": "What is in this image?"},
                   {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
               ]
           }
       ]

       client = LLMClient(model="openai/gpt-4o")
       result, stats = await client.chat_completions(
           messages=messages,
           response_model=ImageDescription
       )
       print(result.description)

**Demo Output:**

.. code-block:: text

   --- Analyzing image with OpenAI ---
   Description: The image shows the Kaval.AI logo, which consists of a stylized geometric 'K' inside a dark circle.
   Objects: Logo, 'K' symbol, Circle
   Colors: White, Dark Blue/Gray
   Stats: Tokens: 450, Cost: $0.0025
