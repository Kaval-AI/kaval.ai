=============
Observability
=============

A pipeline you cannot see into is a pipeline you cannot trust. Kaval.AI makes
every run **observable** by default: each run carries its own state and trace,
every node and model call is logged, and the whole history is persisted so you
can reload and inspect it later — in code or in the backoffice UI.

This guide covers the *why*; for a hands-on tour of storage — chat history,
context, sessions, the tables and writing your own backend — see the
:doc:`../tutorials/observability_storage` tutorial.

What a run records
------------------

Every run produces a :class:`~kavalai.WorkflowState` (see :doc:`workflows`). For
observability the key fields are:

* ``trace`` — the ordered list of visited node names, i.e. the exact path the
  run took through the graph.
* ``token_usage`` — a roll-up of ``model_calls``, ``prompt_tokens``,
  ``completion_tokens``, and ``total_tokens``.
* ``run_id`` / ``session_id`` / ``invocation_id`` — identifiers that tie logs,
  storage, and chat history together. The 8-char ``invocation_id`` prefixes
  every log line of the run, so logs are easy to grep per run.

The state is checkpointed to storage after every node, so even a partial run is
recoverable.

Two pluggable interfaces
------------------------

Persistence and logging are split into two interfaces you can swap:

* **DataStorage** — agents, sessions, runs, and chat history.
* **TaskLogger** — per-node logs and model call stats.

Both are handed to the engine when you build it:

.. code-block:: python

   from kavalai import WorkflowEngine

   engine = WorkflowEngine.from_yaml(yaml, storage=..., task_logger=...)

Local vs. production backends
-----------------------------

For local development and tests, use the SQLite-backed pair:
``SqliteDataStorage`` and ``SqliteTaskLogger``. In production, Postgres backends
map onto the existing ``agents`` / ``sessions`` / ``runs`` / ``chat_messages``
tables and the ``tasks`` / ``model_call_stats`` tables. The same code runs
against either — only the backend changes.

DataStorage
-----------

``DataStorage`` exposes ``initialize_run``, ``update_run``, ``save_state``,
``load_state``, ``add_chat_message``, and ``get_chat_history``. For example, to
reload a finished run or pull back the conversation:

.. code-block:: python

   state = await engine.storage.load_state(state.run_id)
   history = await engine.storage.get_chat_history(state.session_id)

Per-model-call statistics come from the LLM clients themselves: every call
produces a ``ModelCallStat`` with token usage and timing, delivered through the
``ModelStatsReceiver`` callback interface (``ModelStatsLogger`` simply logs
them). See :doc:`../tutorials/llm_clients`.

TaskLogger and fire-and-forget
------------------------------

``TaskLogger`` exposes ``log_node``, ``log_model_call``, ``flush``, and
``close``. Logging is **fire-and-forget** — writes happen in the background so
they never block a run. When you need the writes to land (e.g. at the end of a
test or a batch), await them explicitly:

.. code-block:: python

   await tasklog.flush()

The backoffice UI
-----------------

All of this surfaces in the backoffice as **Conversations -> Runs -> Tasks**,
plus **Metrics** and **Model Calls** pages — letting you drill from a
conversation down to an individual node or model call. See :doc:`../ui/index`.
