Observability & storage
========================

Kaval.AI persists what your agents do, so you can give them **memory**, **reload**
and inspect past runs, and **browse** everything in the backoffice UI. Two small,
pluggable interfaces handle it:

* ``DataStorage`` — agents, sessions, runs, the serialized run state and chat
  history.
* ``TaskLogger`` — per-node debug data and per-call model statistics.

This tutorial focuses on storage: how a conversation is recorded, how a workflow
carries data between steps, what each table holds, and how to swap the backend.
For the *why* behind it, read the :doc:`../guides/observability` guide.

Chat history at the client level
--------------------------------

LLM clients are **stateless** — each call sees exactly the messages you give it.
A single ``prompt`` sends one message; to hold a conversation you build a
``ChatHistory`` (an ordered list of ``ChatMessage``) and append to it yourself:

.. code-block:: python

   from kavalai import OpenAIClient, ChatHistory, ChatMessage

   client = OpenAIClient("gpt-5.4-mini")

   history = ChatHistory(messages=[
       ChatMessage(role="system", content="You are a terse assistant."),
       ChatMessage(role="user", content="My name is Ada."),
       ChatMessage(role="assistant", content="Noted, Ada."),
       ChatMessage(role="user", content="What is my name?"),
   ])
   print(await client.chat_completions(chat_history=history))  # -> "Ada."

Managing that list by hand gets tedious quickly. Workflows do it for you: the
engine writes each turn to storage and replays the conversation on the next one,
which is what gives a chatbot memory across turns (the ``use_history`` flag, on
by default for LLM nodes). The rest of this page is about where those turns — and
everything else — are kept.

Context: how a workflow carries data
------------------------------------

Within a single run a workflow passes data between nodes through its **context**
— a dictionary the engine fills as it goes. The input lands under ``input``; each
node writes its result under its ``output`` name; later nodes read it back by
**context path**, either as a node input or inside a prompt:

.. code-block:: python

   from pydantic import BaseModel
   from kavalai.workflow import WorkflowBuilder, InMemoryDataStorage

   class Email(BaseModel):
       text: str

   class Analysis(BaseModel):
       category: str

   class Reply(BaseModel):
       agent_response: str

   engine = (
       WorkflowBuilder("Triage", llm_model="openai/gpt-5.4-mini")
       .data_model("input", Email)
       .data_model("analysis", Analysis)
       .data_model("output", Reply)
       .start("classify")
       # writes its result into the context under `analysis`
       .llm("classify", prompt="Classify the email as billing, support or other.",
            inputs={"email": "input"}, output="analysis", next="reply")
       # reads it back by context path — here inside the prompt
       .llm("reply", prompt="Write a reply for a {{ context.analysis.category }} email.",
            inputs={"email": "input"}, output="output", next="end")
       .end()
       .build_engine(storage=InMemoryDataStorage())
   )

When the run finishes, the whole context is returned as
:class:`~kavalai.WorkflowState`'s ``data`` and persisted to the run's ``context``
column, so you can reload exactly what each step saw.

Sessions and runs
-----------------

A **session** is one conversation between a user and an agent. Every message the
user sends triggers a **run** — a single invocation of the workflow that produces
one reply. Reuse the same ``session_id`` across turns and all those runs (and
their chat messages) belong to one thread — which is how memory works:

.. code-block:: python

   state1 = await engine.run({"text": "Hi, I'm Ada."}, session_id="user-42")
   state2 = await engine.run({"text": "What's my name?"}, session_id="user-42")

Pass no ``session_id`` and the engine starts a fresh session each time — a
one-off interaction.

The storage backends
--------------------

Storage is pluggable: hand a backend to the engine and the same workflow code
runs against any of them. Kaval.AI ships three.

**InMemoryDataStorage** keeps everything in plain Python structures — no
database, no background thread. It is ephemeral (it lives only as long as the
process or browser page) and perfect for tests, demos and the in-browser
playground. You can drive the storage interface directly; this runs **in your
browser**:

.. code-block:: python
   :class: run-in-browser

   from kavalai.workflow import InMemoryDataStorage

   storage = InMemoryDataStorage()

   # A run belongs to a session. Reuse the session id and the conversation
   # accumulates under it.
   h1 = await storage.initialize_run(workflow_name="Greeter", session_id="user-42")
   await storage.add_chat_message(agent_id=h1.agent_id, session_id=h1.session_id,
                                  run_id=h1.run_id, role="user", content="My name is Ada.")
   await storage.add_chat_message(agent_id=h1.agent_id, session_id=h1.session_id,
                                  run_id=h1.run_id, role="assistant", content="Hi Ada!")

   h2 = await storage.initialize_run(workflow_name="Greeter", session_id="user-42")
   await storage.add_chat_message(agent_id=h2.agent_id, session_id=h2.session_id,
                                  run_id=h2.run_id, role="user", content="What's my name?")

   # Same agent + session across both runs; two different runs.
   print("same agent  :", h1.agent_id == h2.agent_id)
   print("same session:", h1.session_id == h2.session_id)
   print("two runs    :", h1.run_id != h2.run_id)

   for m in await storage.get_chat_history("user-42"):
       print(f"{m.role:>9}: {m.content}")

**SqliteDataStorage** keeps the same data in a local SQLite database — ``:memory:``
by default, or pass a path to persist across process runs:

.. code-block:: python

   from kavalai.workflow import SqliteDataStorage

   storage = SqliteDataStorage("kavalai.db")   # or SqliteDataStorage() for in-memory

**PostgresDataStorage** is the production backend. It writes your agents' data
into the standard ``agents`` / ``sessions`` / ``runs`` / … tables (the same shape
as the other backends) of whatever Postgres database you point it at:

.. code-block:: python

   from kavalai import db_manager
   from kavalai.workflow.storage import PostgresDataStorage

   session_maker = db_manager.get_sessionmaker(
       uri="postgresql://user:pass@localhost:5432/kavalai"
   )
   storage = PostgresDataStorage.from_session_maker(session_maker)

Hand any of them to the engine the same way — only the backend changes:

.. code-block:: python

   from kavalai.workflow import WorkflowEngine

   engine = WorkflowEngine.from_yaml(workflow_yaml, storage=storage)

The tables
----------

Every backend stores the same shape — SQLite and the in-memory store mirror the
Postgres tables column for column — so a run looks identical wherever it lives:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Table
     - What it holds
   * - ``agents``
     - One row per workflow/agent: its name, description, input/output schemas and
       the workflow definition.
   * - ``sessions``
     - One row per conversation, linked to an agent. An optional ``external_id``
       ties it to your own user, ticket or thread id.
   * - ``runs``
     - One row per workflow invocation: the ``input_data``, the ``output_data``
       and the full serialized ``context`` (the checkpointed run state).
   * - ``chat_messages``
     - The conversation turns (``role`` + ``content``) — the basis of chat history.
   * - ``tasks``
     - Per-node debug data: each node's inputs and output, for drilling into what a
       step actually did.
   * - ``model_call_stats``
     - One row per LLM or embedding call: model, token counts, duration and cost.

The first four come from ``DataStorage``; ``tasks`` and ``model_call_stats`` come
from ``TaskLogger`` (the SQLite pair is ``SqliteDataStorage`` +
``SqliteTaskLogger``). The relationship is a simple hierarchy:
**agent → sessions → runs → (chat_messages, tasks, model_call_stats)**.

Bring your own backend
----------------------

Need Redis, MongoDB, a managed store, or your existing database? Implement the
``DataStorage`` interface — six async methods — and pass an instance to the
engine. Nothing else changes:

.. code-block:: python

   from typing import Optional
   from kavalai import DataStorage, RunHandle, ChatMsg, WorkflowState

   class MyStorage(DataStorage):
       async def initialize_run(self, *, workflow_name, description=None,
                                input_schema=None, output_schema=None, workflow=None,
                                session_id=None, external_id=None, input_data=None) -> RunHandle:
           # Create or reuse the agent + session, start a run, and return their ids.
           ...

       async def update_run(self, run_id, *, output_data=None, context=None) -> None:
           ...

       async def save_state(self, run_id, state: WorkflowState) -> None:
           ...  # checkpoint the serialized state (called after every node)

       async def load_state(self, run_id) -> Optional[WorkflowState]:
           ...

       async def add_chat_message(self, *, agent_id, session_id, run_id, role, content) -> None:
           ...

       async def get_chat_history(self, session_id, limit=50) -> list[ChatMsg]:
           ...

       # close() is optional — override it if your backend holds resources.

Today Kaval.AI ships the in-memory, SQLite and Postgres backends, and **more are
planned**. Because the engine only ever talks to the ``DataStorage`` and
``TaskLogger`` interfaces, adding one is purely a matter of implementing the
interface — no engine changes required.

Browsing it in the backoffice
-----------------------------

The **backoffice UI** is a separate service with its own database (it does not
share your agents' tables). It browses agent databases through **projects**: you
register a database — host, port and schema — as a project, and the backoffice
connects to it to show its sessions, runs, tasks and model calls. You can
register several (local, staging, production) behind one UI.

So point ``PostgresDataStorage`` at a database, add that database as a project,
and every session, run, task and model call becomes browsable — drill from a
*conversation* down to an individual *run*, *node* or *model call*, with token
and cost metrics along the way. See :doc:`../ui/index`.

Where to next
-------------

* :doc:`../guides/observability` — the concepts behind observability.
* :doc:`../ui/index` — browse sessions and interactions in the backoffice.
* :doc:`workflow` — the workflows whose runs all of this records.
