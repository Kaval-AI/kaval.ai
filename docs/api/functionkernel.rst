Function Kernel API
===================

The :class:`~kavalai.FunctionKernel` is the unified tool-execution layer. It
hosts three kinds of tools behind a single interface, each addressed by URI:

* **Python functions** — ``python://<name>``
* **REST endpoints** — ``rest://<server>.<tool>``
* **MCP tools** — ``mcp://<server>.<tool>``

Every tool has a Pydantic input and output model; the kernel validates and
coerces arguments before a call and the return value after it, so callers always
see well-typed data.

.. automodule:: kavalai.functionkernel
