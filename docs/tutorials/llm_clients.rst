LLM Clients Tutorial
===================

This tutorial covers the usage of the LLM clients in Kaval.AI SDK.
While Kaval.AI LLM clients are designed to be used internally with the SDK, they are still useful
abstractions for common LLM tasks.

Kaval.AI currently only supports models provideed by OpenAI and Gemini.

Simple Usage
------------

The easiest way to use LLM clients is through the `kavalai.llm_clients.llm_client` module. It provides high-level functions that handle client initialization, retries, and statistics collection.

Basic Prompt with Pydantic Response
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Kaval.AI uses Pydantic models to ensure structured responses from LLMs.

.. code-block:: python

   import asyncio
   from pydantic import BaseModel
   from kavalai.llm_clients.llm_client import chat_completions

   class Greeting(BaseModel):
       message: str
       language: str

   async def main():
       messages = [
           {"role": "user", "content": "Say hello in French"}
       ]

       # Use OpenAI
       result, stats = await chat_completions(
           model="openai/gpt-4o",
           messages=messages,
           response_model=Greeting
       )
       print(f"OpenAI: {result.message} ({result.language})")

       # Use Gemini
       result, stats = await chat_completions(
           model="gemini/gemini-2.0-flash",
           messages=messages,
           response_model=Greeting
       )
       print(f"Gemini: {result.message} ({result.language})")

   if __name__ == "__main__":
       asyncio.run(main())

Computing Embeddings
--------------------

You can compute embeddings for a list of strings using the `compute_embeddings` function.

.. code-block:: python

   from kavalai.llm_clients.llm_client import compute_embeddings

   async def get_embeddings():
       texts = ["Kaval.AI is an agent management system", "LLMs are powerful"]

       embeddings, stats = await compute_embeddings(
           model="openai/text-embedding-3-small",
           texts=texts
       )
       print(f"Computed {len(embeddings)} embeddings")
       return embeddings

Using Images (Multimodal)
-------------------------

Both OpenAI and Gemini clients support multimodal inputs. You can provide images as base64 strings in the `messages` list.

.. code-block:: python

   from kavalai.llm_clients.llm_client import chat_completions
   from pydantic import BaseModel

   class ImageDescription(BaseModel):
       description: str
       objects_found: list[str]

   async def describe_image(image_base64: str):
       messages = [
           {
               "role": "user",
               "content": "What is in this image?",
               "images": [image_base64]
           }
       ]

       result, stats = await chat_completions(
           model="openai/gpt-4o",
           messages=messages,
           response_model=ImageDescription
       )
       print(result.description)

Generating Images
-----------------

Kaval.AI also supports image generation via DALL-E (OpenAI) and Imagen (Gemini).

.. code-block:: python

   from kavalai.llm_clients.llm_client import generate_image

   async def create_art():
       # Generate with OpenAI DALL-E 3
       img_base64, stats = await generate_image(
           model="openai/dalle-3",
           prompt="A futuristic city with flying cars in cyberpunk style",
           size="1024x1024"
       )

       # Generate with Gemini Imagen
       img_base64, stats = await generate_image(
           model="gemini/imagen-3.0-generate-002",
           prompt="A serene landscape with mountains and a lake at sunset"
       )

       # The returned img_base64 can be saved to a file or displayed
