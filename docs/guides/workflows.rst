==========
Workflows
==========

A **workflow** is the backbone of Kaval.AI: a directed graph — a small state
machine — that turns an input into an output by walking from a ``start`` node to
an ``end`` node. Workflows make agentic pipelines *predictable*: every step is
named, every edge is explicit, and every value that crosses a node boundary is
typed and validated.

You author a workflow in two equivalent ways: declaratively in **YAML**, or
programmatically with the fluent :class:`~kavalai.WorkflowBuilder`. Both compile
to the same :class:`~kavalai.WorkflowGraph` and run on the same
:class:`~kavalai.WorkflowEngine`.

For a hands-on walkthrough, see :doc:`../tutorials/workflow`.

The graph model
---------------

A workflow is just three things:

* a **name**,
* a set of **data_types**, and
* a set of **nodes**.

The caller hands an input to the ``start`` node and reads the result off the
``end`` node. By convention the input is the data type named ``input``, and the
returned value is the ``output`` variable named on the ``end`` node.

Data types and validation
--------------------------

``data_types`` are JSON-schema fragments that Kaval.AI compiles into Pydantic
models. Because every node input and output is described by one of these models,
the engine validates data at each boundary — a node can never silently receive
or emit a malformed value. This typed-I/O guarantee is one of the pillars of
:doc:`safety` in Kaval.AI.

Node types
----------

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Node
     - What it does
   * - ``start``
     - Entry point; receives the workflow input.
   * - ``end``
     - Exit point; names the ``output`` variable to return.
   * - ``llm``
     - One structured LLM completion.
   * - ``agent``
     - A multi-step, tool-using agent loop bounded by ``max_steps`` (see :doc:`agents`).
   * - ``function``
     - A single tool call through the FunctionKernel, addressed by URI (see :doc:`tools`).
   * - ``if``
     - Branches on a boolean ``condition``.
   * - ``switch``
     - Evaluates ``expr``, stringifies it, matches against ``cases``, else ``default``.

The corresponding node classes — :class:`~kavalai.StartNode`,
:class:`~kavalai.EndNode`, :class:`~kavalai.LLMNode`,
:class:`~kavalai.AgentNode`, :class:`~kavalai.FunctionNode`,
:class:`~kavalai.IfNode`, :class:`~kavalai.SwitchNode` — are importable from the
top-level :mod:`kavalai` package.

Context and interpolation
--------------------------

As a workflow runs, values accumulate in a shared **context**. Inside a prompt
you interpolate from it with ``{{ context.<path> }}``. A node input is written
as ``{type: context, value: <path>}``; in the :class:`~kavalai.WorkflowBuilder`
a bare string input is treated as a context path. Branching nodes (``if`` /
``switch``) read the context through the safe expression language described in
:doc:`safety`.

YAML vs. WorkflowBuilder
------------------------

The :class:`~kavalai.WorkflowBuilder` mirrors the YAML structure with chainable
methods — ``data_type``, ``start``, ``end``, ``llm``, ``agent``, ``function``,
``if_``, ``switch`` — each returning ``self``. Finish with ``build()`` for a
:class:`~kavalai.WorkflowGraph` or ``build_engine()`` for a ready
:class:`~kavalai.WorkflowEngine`.

To load and run from YAML:

.. code-block:: python

   from kavalai import WorkflowEngine

   engine = WorkflowEngine.from_yaml(yaml, storage=..., task_logger=...)
   state = await engine.run({...})

Other constructors include ``WorkflowEngine.from_yaml_path`` and
``WorkflowEngine.from_dict``.

The WorkflowState
-----------------

Every run produces a :class:`~kavalai.WorkflowState`, which is both the result
and the audit trail:

* ``status`` — terminal state of the run.
* ``trace`` — ordered list of visited node names.
* ``data`` — the full context.
* ``input_data`` / ``output_data`` — the values in and out.
* ``run_id`` / ``session_id`` / ``invocation_id`` — identifiers; the 8-char
  ``invocation_id`` prefixes every log line of the run.
* ``token_usage`` — ``model_calls``, ``prompt_tokens``, ``completion_tokens``,
  ``total_tokens``.

The state is serialisable via ``state.to_json()`` and
``WorkflowState.from_json()``, and it is checkpointed to storage after every
node — so a run can be reloaded and inspected later. See :doc:`observability`
for how this powers the backoffice UI.
