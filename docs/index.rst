.. image:: _static/iconlogo.svg
   :width: 300
   :align: left
   :alt: Kaval.AI

**Kaval.AI is a Python framework for building agentic AI pipelines that are
predictable, observable, and safe.**

You describe an agent as a *workflow* — a small, typed state machine of nodes
(LLM calls, tool calls, multi-step agents, and branches). The engine runs it as
a single interaction, validating every step against your schemas, checkpointing
a serialisable state, and recording per-node debug data and model statistics.
Author workflows in YAML or with the fluent :class:`~kavalai.WorkflowBuilder`,
run them from the SDK, and manage and monitor them in the backoffice UI.

.. code-block:: python

   from kavalai import WorkflowEngine

   engine = WorkflowEngine.from_yaml(greeter_yaml)
   state = await engine.run({"user_message": "Hi, I'm Timo!"})

   print(state.output_data["agent_response"])
   print(state.trace)         # ['start', 'reply', 'end']
   print(state.token_usage)   # {'model_calls': 1, 'total_tokens': 106, ...}

Why Kaval.AI
------------

.. rubric:: Predictable

Workflows are **declarative typed graphs**. ``data_types`` are JSON-schema
fragments compiled to Pydantic models, so every node's input and output is
validated. Branches route on a **safe expression language** (an AST whitelist,
never ``eval``), and you can inject a stub client to get fully deterministic,
offline runs for testing.

.. rubric:: Observable

Every run returns a JSON-serialisable :class:`~kavalai.WorkflowState` with the
node ``trace``, full context, and aggregate ``token_usage``. Pluggable
**storage** and **task-logger** backends checkpoint state after every node and
record per-node timing and per-call token/cost statistics — surfaced in the
backoffice UI as conversations, runs, tasks, metrics and model calls.

.. rubric:: Safe

Tools run through one validated interface (the :class:`~kavalai.FunctionKernel`),
agents are bounded by an explicit ``max_steps`` budget and tool allow-lists, and
structured outputs are coerced into the models you define. Nothing reaches an LLM
or a tool without passing through a schema first.

Get started
-----------

.. toctree::
   :maxdepth: 2
   :caption: Learn

   tutorials/index
   guides/index

.. toctree::
   :maxdepth: 2
   :caption: Operate

   ui/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
