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

from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel

from kavalai.workflow.state import WorkflowState


class RunHandle(BaseModel):
    """Identifiers created when a workflow run is initialized.

    Mirrors the agent/session/run triple of the existing persistence layer so
    a Postgres backend can map straight onto ``AgentService``.
    """

    agent_id: str
    session_id: str
    run_id: str


class ChatMsg(BaseModel):
    """A single chat message row returned from history queries."""

    role: str
    content: Optional[str] = None


class DataStorage(ABC):
    """Common data-storage interface for workflow persistence.

    Backends implement this to persist agents, sessions, runs, chat messages
    and the serialized :class:`WorkflowState`. The method shapes intentionally
    mirror :class:`~kavalai.agent_service.AgentService` so a Postgres
    backend is a thin delegation layer over the existing tables.
    """

    @abstractmethod
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
        """Create (or reuse) the agent + session and start a new run."""

    @abstractmethod
    async def update_run(
        self,
        run_id: str,
        *,
        output_data: Optional[dict] = None,
        context: Optional[dict] = None,
    ) -> None:
        """Persist the final output and/or context of a run."""

    @abstractmethod
    async def save_state(self, run_id: str, state: WorkflowState) -> None:
        """Checkpoint the serialized workflow state for a run."""

    @abstractmethod
    async def load_state(self, run_id: str) -> Optional[WorkflowState]:
        """Load the last checkpointed workflow state for a run, if any."""

    @abstractmethod
    async def add_chat_message(
        self,
        *,
        agent_id: str,
        session_id: str,
        run_id: Optional[str],
        role: str,
        content: Optional[str],
    ) -> None:
        """Append a chat message to a session."""

    @abstractmethod
    async def get_chat_history(self, session_id: str, limit: int = 50) -> list[ChatMsg]:
        """Return the chat history for a session, oldest first."""

    async def close(self) -> None:
        """Release any resources held by the backend. Override if needed."""
        return None
