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
from typing import Any, Optional
from uuid import uuid4

import aiosqlite

from kavalai.agents.utils import to_plain
from kavalai.agents.v2.workflow.tasklog.base import TaskLogger
from kavalai.llm_clients.base_client import ModelCallStat

# Mirrors the Postgres ``tasks`` and ``model_call_stats`` tables (db.py) using
# TEXT UUIDs and JSON-encoded payload columns.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    agent_id TEXT,
    session_id TEXT,
    run_id TEXT,
    name TEXT,
    node_type TEXT,
    inputs TEXT,
    output TEXT,
    prompt TEXT,
    errors TEXT,
    duration_seconds REAL
);
CREATE TABLE IF NOT EXISTS model_call_stats (
    id TEXT PRIMARY KEY,
    call_type TEXT NOT NULL,
    model TEXT,
    agent_id TEXT,
    request_data TEXT,
    response_data TEXT,
    response_code INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    batch_size INTEGER,
    duration_seconds REAL
);
"""


def _dumps(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(to_plain(value))


class SqliteTaskLogger(TaskLogger):
    """Task logger storing node executions and model stats in SQLite.

    Defaults to a private ``:memory:`` database. Pass a file ``path`` to keep
    the debugging data across runs.
    """

    def __init__(self, path: str = ":memory:"):
        super().__init__()
        self.path = path
        self._conn: Optional[aiosqlite.Connection] = None

    async def _connect(self) -> aiosqlite.Connection:
        if self._conn is None:
            self._conn = await aiosqlite.connect(self.path)
            self._conn.row_factory = aiosqlite.Row
            await self._conn.executescript(_SCHEMA)
            await self._conn.commit()
        return self._conn

    async def _log_node_impl(
        self,
        *,
        run_id: Optional[str],
        session_id: Optional[str],
        agent_id: Optional[str],
        node_name: str,
        node_type: str,
        inputs: Optional[dict],
        output: Any,
        prompt: Optional[str],
        duration: float,
        errors: Optional[list[str]],
    ) -> None:
        conn = await self._connect()
        await conn.execute(
            "INSERT INTO tasks (id, agent_id, session_id, run_id, name, "
            "node_type, inputs, output, prompt, errors, duration_seconds) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid4()),
                agent_id,
                session_id,
                run_id,
                node_name,
                node_type,
                _dumps(inputs),
                _dumps(output),
                prompt,
                _dumps(errors),
                duration,
            ),
        )
        await conn.commit()

    async def _log_model_call_impl(
        self, stats: ModelCallStat, agent_id: Optional[str]
    ) -> None:
        conn = await self._connect()
        await conn.execute(
            "INSERT INTO model_call_stats (id, call_type, model, agent_id, "
            "request_data, response_data, response_code, prompt_tokens, "
            "completion_tokens, total_tokens, batch_size, duration_seconds) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid4()),
                stats.call_type,
                stats.model,
                agent_id,
                stats.request_data,
                stats.response_data,
                stats.response_code,
                stats.prompt_tokens,
                stats.completion_tokens,
                stats.total_tokens,
                stats.batch_size,
                stats.duration_seconds,
            ),
        )
        await conn.commit()

    async def close(self) -> None:
        await self.flush()
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
