Agents API
==========

The agent runtime lives in :mod:`kavalai.agents`. The headline class is
:class:`~kavalai.agents.agent.Agent` — a multi-step reasoning loop that calls
tools through a :class:`~kavalai.FunctionKernel` until it produces a final,
optionally structured, answer.

Agent
-----

.. automodule:: kavalai.agents.agent

Run context
-----------

.. automodule:: kavalai.agents.run_context

Workflow configuration models
-----------------------------

.. automodule:: kavalai.agents.workflow_model

Agent service & persistence
---------------------------

.. automodule:: kavalai.agent_service

.. automodule:: kavalai.agents.sessions

Remote agent client
--------------------

.. automodule:: kavalai.agents.client

RAG service
-----------

.. automodule:: kavalai.agents.rag_service
   :exclude-members: build_batch_query_cte, batch_query_with_join
