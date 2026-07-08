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

``HistoryService`` — storage abstraction for the agent runtime's history data:
chat messages, task execution records, and model-call statistics.

This is the swap point for custom history storage. The built-in SQL backend
(:class:`kavalai.history.sql.SqlHistoryService`) stores history in the agents
migration set's tables; a custom backend can send it anywhere (its storage is
then entirely its own concern — the migration set does not apply to it).

History records reference core entities (agents, sessions, runs) by UUID. The
built-in SQL backend additionally enforces FK constraints, but that is a
private detail of that backend — the interface only promises the ids.

Unlike :class:`~kavalai.workflow.tasklog.base.TaskLogger` (fire-and-forget
write-behind logging), chat history has read paths the runtime depends on:
the conversation context is read back on every run.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID


class HistoryService(ABC):
    """
    Interface for agent history storage backends.

    Returned records are backend-specific objects; the interface guarantees
    the *fields* named in each method's docstring are available as attributes
    (e.g. a chat message record exposes ``role``, ``content``, ``created_at``).
    """

    # -- chat history --------------------------------------------------------

    @abstractmethod
    async def add_chat_message(
        self,
        agent_id: UUID,
        session_id: UUID,
        role: str,
        content: Optional[str],
        run_id: Optional[UUID] = None,
    ):
        """Append a message to a session's chat history.

        Returns a record with at least: id, agent_id, session_id, run_id,
        role, content, created_at.
        """

    @abstractmethod
    async def get_chat_messages(self, session_id: UUID, limit: int = 50) -> List:
        """Return a session's chat history ordered oldest-to-newest.

        ``limit`` caps the number of messages returned (windowing for prompt
        construction). Records expose at least: role, content, created_at.
        """

    # -- task records --------------------------------------------------------

    @abstractmethod
    async def add_task(
        self,
        session_id: UUID,
        run_id: UUID,
        name: Optional[str] = None,
        agent_id: Optional[UUID] = None,
        inputs: Optional[Dict] = None,
        output: Optional[Dict] = None,
        prompt: Optional[str] = None,
        errors: Optional[list[str]] = None,
        duration_seconds: Optional[float] = None,
        node_type: Optional[str] = None,
    ):
        """Record one unit of work (workflow-node execution) within a run."""

    # -- model call stats ----------------------------------------------------

    @abstractmethod
    async def add_model_call_stat(self, stats: Any, agent_id: Optional[UUID] = None):
        """Record LLM/embedding call statistics (a ``ModelCallStat``)."""

    @abstractmethod
    async def get_model_call_stats(
        self,
        call_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List:
        """Paginated model-call stats, newest first, optionally by call type."""

    # -- deletion ------------------------------------------------------------
    #
    # Explicit deletion semantics: custom backends have no FK cascades from
    # the core tables, so callers use these when a session/agent goes away.

    @abstractmethod
    async def delete_for_session(self, session_id: UUID) -> None:
        """Delete all history (chat, tasks) belonging to a session."""

    @abstractmethod
    async def delete_for_agent(self, agent_id: UUID) -> None:
        """Delete all history (chat, tasks, stats) belonging to an agent."""

    async def delete_for_sessions(self, session_ids: Sequence[UUID]) -> None:
        """Bulk variant of :meth:`delete_for_session`."""
        for session_id in session_ids:
            await self.delete_for_session(session_id)
