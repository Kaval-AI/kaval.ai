======
Agents
======

An :class:`~kavalai.Agent` is a lightweight, multi-step reasoning loop: give it
a task and a set of tools, and it works toward an answer one step at a time. The
loop is deliberately small and explicit — bounded by ``max_steps`` and driven by
tools addressed by URI — so that agentic behaviour stays predictable and
auditable rather than open-ended.

For a hands-on walkthrough, see :doc:`../tutorials/agents`.

Construction
------------

.. code-block:: python

   from kavalai import Agent, OpenAIClient, FunctionKernel

   agent = Agent(
       llm_client=OpenAIClient("gpt-4.1-mini"),
       kernel=FunctionKernel(),   # optional
       run_context=...,           # optional
       prompt_template=...,       # optional Jinja2 Template
       debug=False,
   )

Any provider client works, since they share the :class:`BaseLlmClient`
interface — :class:`~kavalai.OpenAIClient`, :class:`~kavalai.GeminiClient`, or
:class:`~kavalai.OllamaClient` (e.g. ``GeminiClient("gemini-2.5-flash")``). See
:doc:`../tutorials/llm_clients`.

Running a prompt
----------------

.. code-block:: python

   result = await agent.prompt(
       "Summarise the latest filings",
       response_model=MySchema,   # optional Pydantic model
       max_steps=10,
   )

When you pass a ``response_model`` the agent returns an instance of it; without
one it returns a plain string.

The four-step cycle
-------------------

Each step of the loop does the same four things:

#. **Render** a system prompt from the Jinja2 template, including the task, the
   available tool descriptions, and the history of previous steps.
#. **Reason** — the LLM returns a ``StepOutput``: a list of ``tool_calls`` plus
   an optional final output.
#. **Act** — the requested tool calls execute *in parallel* through the
   :class:`~kavalai.FunctionKernel`, and their results feed into the next step.
#. **Decide** — the loop stops when the model returns output with no further
   tool calls, or when ``max_steps`` is reached.

This explicit bound is a safety feature: an agent can never loop forever, and it
can only act through tools you have registered (see :doc:`tools` and
:doc:`safety`).

Structured output
-----------------

Because the final answer can be validated against a ``response_model``, an agent
slots cleanly into a typed pipeline — its output is just another typed value,
the same as any other node boundary.

The ``agent`` workflow node
---------------------------

The ``agent`` node in a workflow graph runs this exact same loop inside the
graph, with its own ``max_steps``. So you can drop an agent into a larger,
deterministic :doc:`workflow <workflows>` and still get the per-node trace and
token accounting described in :doc:`observability`.
