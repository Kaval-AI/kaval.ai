LLM Clients API
===============

:mod:`kavalai.llm_clients` provides a unified, observable interface over LLM and
embedding providers. Every call returns a :class:`~kavalai.ModelCallStat` with
token usage and timing, and structured output is validated against a Pydantic
``response_model``.

Base client and models
----------------------

.. automodule:: kavalai.llm_clients.base_client

Provider clients
----------------

.. automodule:: kavalai.llm_clients.openai_client

.. automodule:: kavalai.llm_clients.gemini_client

.. automodule:: kavalai.llm_clients.ollama_client

Embeddings
----------

.. automodule:: kavalai.llm_clients.embeddings

Streaming
---------

.. automodule:: kavalai.llm_clients.streamer
