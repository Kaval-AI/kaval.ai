Getting Started
===============

This guide gets you from an empty environment to a running agent in a few
minutes. You will install Kaval.AI, configure an LLM provider, and run your
first **workflow** — a small typed state machine that takes an input and
returns a structured result.

.. admonition:: Run it in your browser 🐍
   :class: tip

   Every Python snippet in these docs has a **Run in browser ▶** button. It
   boots `Pyodide <https://pyodide.org>`_, installs Kaval.AI client-side and runs
   the code in a side panel — nothing to install locally. The snippet below
   needs no API key, so try it first:

   .. code-block:: python

      from kavalai import evaluate_expression, evaluate_bool, FunctionKernel, pythontool

      # 1) Pure-Python expression engine
      ctx = {"user": {"name": "Ada", "age": 36}}
      print("name     =", evaluate_expression("user.name", ctx))
      print("is_adult =", evaluate_bool("user.age >= 18", ctx))

      # 2) Register a tool and call it through the FunctionKernel
      @pythontool
      def add(a: int, b: int) -> int:
          return a + b

      kernel = FunctionKernel()
      kernel.register_python_tool("add", add)
      print("add(2, 40) =", await kernel.call_tool("python://add", {"a": 2, "b": 40}))

   To run the LLM workflows below, open the panel's **API keys** section and
   paste an OpenAI or Gemini key — it is stored only in your browser. Note that
   provider calls made directly from the browser may be blocked by CORS.

Installation
------------

Kaval.AI targets **Python 3.12+**. Install it into a virtual environment.

With ``uv`` (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   uv venv
   uv pip install kavalai

With ``pip``
^^^^^^^^^^^^

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install kavalai

Optional extras enable extra capabilities:

* ``kavalai[tools]`` — the Crawl4AI web-scraping tool.
* ``kavalai[notebooks]`` — Jupyter support for running the tutorial notebooks.

Installing from source
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/Kaval-AI/kaval.ai.git
   cd kaval.ai
   uv pip install -e .

Configure a provider
---------------------

LLM nodes call a provider, so set the relevant API key in your environment (or a
``.env`` file). Kaval.AI ships native clients for OpenAI, Google Gemini and
Ollama.

.. code-block:: bash

   export OPENAI_API_KEY="sk-..."
   # or
   export GEMINI_API_KEY="..."

Setting a default model lets you omit ``llm_model`` from your workflows:

.. code-block:: bash

   export KAVALAI_DEFAULT_LLM_MODEL="openai/gpt-4o-mini"

Your first workflow
-------------------

A workflow is ``name`` + ``data_types`` + ``nodes``. The caller hands an input
to the **start** node and reads the result from the **end** node. ``data_types``
are JSON-schema fragments compiled to Pydantic models, so every node's input and
output is validated.

This minimal graph — ``start → llm → end`` — greets the user by name:

.. code-block:: python

   import asyncio
   from kavalai import WorkflowEngine

   greeter_yaml = """
   name: Greeter
   description: A one-node agent that greets the user by name.
   data_types:
     input:
       type: object
       properties:
         user_message: {type: string}
     output:
       type: object
       properties:
         agent_response: {type: string}
   nodes:
     - {name: start, type: start, next: reply}
     - name: reply
       type: llm
       prompt: |
         You are a warm, concise greeter.
         Reply in one friendly sentence: {{ context.input.user_message }}
       inputs:
         input: {type: context, value: input}
       output: output
       next: end
     - {name: end, type: end, output: output}
   """


   async def main():
       engine = WorkflowEngine.from_yaml(greeter_yaml)
       state = await engine.run({"user_message": "Hi, I'm Timo!"})
       print(state.output_data["agent_response"])
       print("path:", " → ".join(state.trace))


   asyncio.run(main())

Running it prints the reply and the path the engine took through the graph:

.. code-block:: text

   Hi Timo! It's great to meet you!
   path: start → reply → end

Every run returns a :class:`~kavalai.WorkflowState`: the status, the ordered
``trace`` of visited nodes, the full context ``data``, the final ``output_data``,
and an aggregate ``token_usage``. It is JSON-serialisable and checkpointed after
every node, so runs can be reloaded and inspected later.

The same graph in code
----------------------

You don't have to write YAML. The fluent :class:`~kavalai.WorkflowBuilder`
produces the identical graph and is handy for generating workflows dynamically:

.. code-block:: python

   from kavalai import WorkflowBuilder

   engine = (
       WorkflowBuilder("Greeter")
       .data_type("input", {"user_message": str})
       .data_type("output", {"agent_response": str})
       .start("reply")
       .llm(
           "reply",
           prompt="Reply in one friendly sentence: {{ context.input.user_message }}",
           inputs={"input": "input"},   # a bare string is treated as a context path
           output="output",
           next="end",
       )
       .end()
       .build_engine()
   )

Where to next
-------------

* :doc:`workflow` — branching, function/agent nodes, observability and
  deterministic testing.
* :doc:`agents` — the ``FunctionKernel`` and the multi-step ``Agent`` loop.
* :doc:`llm_clients` — calling models directly, structured output and streaming.
* :doc:`../guides/index` — the concepts behind the engine.
* :doc:`../ui/index` — manage and monitor agents in the backoffice UI.
