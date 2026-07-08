Workflow API
============

The workflow engine lives in :mod:`kavalai.workflow`. It turns a YAML graph (or a
:class:`~kavalai.WorkflowBuilder` chain) into an executable state machine, and
checkpoints a serialisable state through pluggable storage and task-logger
backends.

Engine and builder
-------------------

.. automodule:: kavalai.workflow.engine

.. automodule:: kavalai.workflow.builder

Graph models
------------

.. automodule:: kavalai.workflow.models

Run state
---------

.. automodule:: kavalai.workflow.state

Expressions
-----------

.. automodule:: kavalai.workflow.expressions

Client factory
--------------

.. automodule:: kavalai.workflow.clients

Task logging backends
---------------------

.. automodule:: kavalai.workflow.tasklog.base

.. automodule:: kavalai.workflow.tasklog.sqlite
