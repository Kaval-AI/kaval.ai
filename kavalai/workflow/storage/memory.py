"""
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
from typing import Optional
from uuid import uuid4

import aiosqlite

from kavalai.workflow.state import WorkflowState
from kavalai.workflow.storage.base import ChatMsg, DataStorage, RunHandle

# Schema mirrors the agents tables (agents/sessions/runs/chat_messages) defined
# by the ORM models in kavalai/agents/db.py, using TEXT UUIDs and JSON-encoded
# columns so the in-memory backend has identical semantics to the production
# backend.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    input_schema TEXT,
    output_schema TEXT,
    workflow TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    external_id TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    input_data TEXT,
    output_data TEXT,
    context TEXT
);
CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    run_id TEXT,
    role TEXT NOT NULL,
    content TEXT,
    seq INTEGER
);
"""


def _dumps(value: Optional[dict]) -> Optional[str]:
    return None if value is None else json.dumps(value)


class SqliteDataStorage(DataStorage):
    """In-memory (or file-backed) data storage using SQLite via aiosqlite.

    Defaults to a private ``:memory:`` database, which makes it ideal for local
    development and tests. Pass a file ``path`` to persist across process runs.
    """

    def __init__(self, path: str = ":memory:"):
        self.path = path
        self._conn: Optional[aiosqlite.Connection] = None

    async def _connect(self) -> aiosqlite.Connection:
        if self._conn is None:
            self._conn = await aiosqlite.connect(self.path)
            self._conn.row_factory = aiosqlite.Row
            await self._conn.executescript(_SCHEMA)
            await self._conn.commit()
        return self._conn

    async def initialize_run(
        self,
        *,
        workflow_name: str,
        description: Optional[str] = None,
        input_schema: Optional[dict] = None,
        output_schema: Optional[dict] = None,
        workflow: Optional[dict] = None,
        session_id: Optional[str] = None,
        external_id: Optional[str] = None,
        input_data: Optional[dict] = None,
    ) -> RunHandle:
        conn = await self._connect()

        # Reuse an agent with the same name, mirroring get_or_create_agent.
        async with conn.execute(
            "SELECT id FROM agents WHERE name = ? LIMIT 1", (workflow_name,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is not None:
            agent_id = row["id"]
        else:
            agent_id = str(uuid4())
            await conn.execute(
                "INSERT INTO agents (id, name, description, input_schema, "
                "output_schema, workflow) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    agent_id,
                    workflow_name,
                    description,
                    _dumps(input_schema),
                    _dumps(output_schema),
                    _dumps(workflow),
                ),
            )

        # Reuse the session when an id is supplied and exists, else create one.
        resolved_session_id: Optional[str] = None
        if session_id is not None:
            async with conn.execute(
                "SELECT id FROM sessions WHERE id = ? LIMIT 1", (session_id,)
            ) as cursor:
                existing = await cursor.fetchone()
            if existing is not None:
                resolved_session_id = session_id
        if resolved_session_id is None:
            resolved_session_id = session_id or str(uuid4())
            await conn.execute(
                "INSERT INTO sessions (id, agent_id, external_id) VALUES (?, ?, ?)",
                (resolved_session_id, agent_id, external_id),
            )

        run_id = str(uuid4())
        await conn.execute(
            "INSERT INTO runs (id, session_id, input_data) VALUES (?, ?, ?)",
            (run_id, resolved_session_id, _dumps(input_data)),
        )
        await conn.commit()

        return RunHandle(
            agent_id=agent_id, session_id=resolved_session_id, run_id=run_id
        )

    async def update_run(
        self,
        run_id: str,
        *,
        output_data: Optional[dict] = None,
        context: Optional[dict] = None,
    ) -> None:
        conn = await self._connect()
        sets = []
        params: list = []
        if output_data is not None:
            sets.append("output_data = ?")
            params.append(_dumps(output_data))
        if context is not None:
            sets.append("context = ?")
            params.append(_dumps(context))
        if not sets:
            return
        params.append(run_id)
        await conn.execute(f"UPDATE runs SET {', '.join(sets)} WHERE id = ?", params)
        await conn.commit()

    async def save_state(self, run_id: str, state: WorkflowState) -> None:
        conn = await self._connect()
        await conn.execute(
            "UPDATE runs SET context = ? WHERE id = ?",
            (state.to_json(), run_id),
        )
        await conn.commit()

    async def load_state(self, run_id: str) -> Optional[WorkflowState]:
        conn = await self._connect()
        async with conn.execute(
            "SELECT context FROM runs WHERE id = ? LIMIT 1", (run_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None or row["context"] is None:
            return None
        return WorkflowState.from_json(row["context"])

    async def add_chat_message(
        self,
        *,
        agent_id: str,
        session_id: str,
        run_id: Optional[str],
        role: str,
        content: Optional[str],
    ) -> None:
        conn = await self._connect()
        async with conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM chat_messages "
            "WHERE session_id = ?",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        seq = row["next_seq"]
        await conn.execute(
            "INSERT INTO chat_messages (id, agent_id, session_id, run_id, role, "
            "content, seq) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(uuid4()), agent_id, session_id, run_id, role, content, seq),
        )
        await conn.commit()

    async def get_chat_history(self, session_id: str, limit: int = 50) -> list[ChatMsg]:
        conn = await self._connect()
        async with conn.execute(
            "SELECT role, content FROM chat_messages WHERE session_id = ? "
            "ORDER BY seq ASC LIMIT ?",
            (session_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [ChatMsg(role=row["role"], content=row["content"]) for row in rows]

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None


class InMemoryDataStorage(DataStorage):
    """Pure-Python, thread-free in-memory data storage.

    A dependency-free alternative to :class:`SqliteDataStorage` that keeps
    everything in plain Python structures. Crucially it spawns no background
    thread, so it works under Pyodide / WebAssembly — where ``aiosqlite`` cannot
    start its worker thread (``RuntimeError: can't start new thread``). It is the
    natural store for the in-browser playground and for fast unit tests.

    Semantics mirror :class:`SqliteDataStorage`: agents are reused by name,
    sessions are reused when a known id is supplied, and chat history is returned
    oldest-first. State lives only for the lifetime of the instance.
    """

    def __init__(self) -> None:
        self._agent_ids_by_name: dict[str, str] = {}
        self._sessions: set[str] = set()
        # run_id -> {"output_data": ..., "context": ...}
        self._runs: dict[str, dict] = {}
        self._states: dict[str, WorkflowState] = {}
        # session_id -> ordered chat messages
        self._chat: dict[str, list[ChatMsg]] = {}

    async def initialize_run(
        self,
        *,
        workflow_name: str,
        description: Optional[str] = None,
        input_schema: Optional[dict] = None,
        output_schema: Optional[dict] = None,
        workflow: Optional[dict] = None,
        session_id: Optional[str] = None,
        external_id: Optional[str] = None,
        input_data: Optional[dict] = None,
    ) -> RunHandle:
        # Reuse an agent with the same name, mirroring get_or_create_agent.
        agent_id = self._agent_ids_by_name.get(workflow_name)
        if agent_id is None:
            agent_id = str(uuid4())
            self._agent_ids_by_name[workflow_name] = agent_id

        # Reuse the session when a known id is supplied, else create one (keeping
        # a supplied-but-unknown id so callers can pin their own session ids).
        if session_id is not None and session_id in self._sessions:
            resolved_session_id = session_id
        else:
            resolved_session_id = session_id or str(uuid4())
            self._sessions.add(resolved_session_id)

        run_id = str(uuid4())
        self._runs[run_id] = {"output_data": None, "context": None}
        return RunHandle(
            agent_id=agent_id, session_id=resolved_session_id, run_id=run_id
        )

    async def update_run(
        self,
        run_id: str,
        *,
        output_data: Optional[dict] = None,
        context: Optional[dict] = None,
    ) -> None:
        run = self._runs.get(run_id)
        if run is None:
            return
        if output_data is not None:
            run["output_data"] = output_data
        if context is not None:
            run["context"] = context

    async def save_state(self, run_id: str, state: WorkflowState) -> None:
        self._states[run_id] = state

    async def load_state(self, run_id: str) -> Optional[WorkflowState]:
        return self._states.get(run_id)

    async def add_chat_message(
        self,
        *,
        agent_id: str,
        session_id: str,
        run_id: Optional[str],
        role: str,
        content: Optional[str],
    ) -> None:
        self._chat.setdefault(session_id, []).append(
            ChatMsg(role=role, content=content)
        )

    async def get_chat_history(self, session_id: str, limit: int = 50) -> list[ChatMsg]:
        return list(self._chat.get(session_id, []))[:limit]
