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

from typing import Optional, Any
import asyncio
from loguru import logger

from kavalai.agents.agent_service import AgentService
from kavalai.agents.run_context import RunContext


class TaskLogger:
    """Component for logging tasks to the database with consistent structure.

    All methods handle errors internally and log them without raising exceptions.
    Uses fire-and-forget pattern to avoid blocking workflow execution.
    """

    def __init__(self, agent_service: Optional[AgentService], run_context: RunContext):
        self.agent_service = agent_service
        self.run_context = run_context
        self._background_tasks = set()

    def _create_background_task(self, coro):
        """Create a background task with proper error handling."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        task.add_done_callback(self._handle_task_exception)
        return task

    def _handle_task_exception(self, task: asyncio.Task):
        """Log exceptions from background tasks."""
        try:
            task.result()
        except Exception as e:
            logger.error(f"Background task logging failed: {e}")

    async def _log_llm_task_impl(
        self,
        task_name: str,
        prompt: str,
        input_data: dict,
        output: Any,
        duration: float,
        errors: Optional[list[str]] = None,
    ):
        """Internal implementation for logging LLM task."""
        await self.agent_service.add_task(
            agent_id=self.run_context.agent_id,
            session_id=self.run_context.session_id,
            run_id=self.run_context.run_id,
            name=task_name,
            prompt=prompt,
            inputs=input_data,
            output=output,
            errors=errors,
            duration_seconds=duration,
        )

    def log_llm_task(
        self,
        task_name: str,
        prompt: str,
        input_data: dict,
        output: Any,
        duration: float,
        errors: Optional[list[str]] = None,
    ):
        """Log an LLM task with full prompt (fire-and-forget)."""
        if not self.agent_service or not self.run_context.run_id:
            return
        self._create_background_task(
            self._log_llm_task_impl(task_name, prompt, input_data, output, duration, errors)
        )

    async def _log_agent_task_impl(
        self,
        task_name: str,
        system_prompt: str,
        input_data: dict,
        output: Any,
        duration: float,
        errors: Optional[list[str]] = None,
    ):
        """Internal implementation for logging agent task."""
        await self.agent_service.add_task(
            agent_id=self.run_context.agent_id,
            session_id=self.run_context.session_id,
            run_id=self.run_context.run_id,
            name=task_name,
            prompt=system_prompt,
            inputs=input_data,
            output=output,
            errors=errors,
            duration_seconds=duration,
        )

    def log_agent_task(
        self,
        task_name: str,
        system_prompt: str,
        input_data: dict,
        output: Any,
        duration: float,
        errors: Optional[list[str]] = None,
    ):
        """Log an agent task with full system prompt (fire-and-forget)."""
        if not self.agent_service or not self.run_context.run_id:
            return
        self._create_background_task(
            self._log_agent_task_impl(task_name, system_prompt, input_data, output, duration, errors)
        )

    async def _log_tool_call_impl(
        self,
        tool_uri: str,
        arguments: dict,
        output: Any,
        duration: float,
        errors: Optional[list[str]] = None,
    ):
        """Internal implementation for logging tool call."""
        await self.agent_service.add_task(
            agent_id=self.run_context.agent_id,
            session_id=self.run_context.session_id,
            run_id=self.run_context.run_id,
            name=tool_uri,
            inputs={"arguments": arguments},
            output=output,
            errors=errors,
            duration_seconds=duration,
        )

    def log_tool_call(
        self,
        tool_uri: str,
        arguments: dict,
        output: Any,
        duration: float,
        errors: Optional[list[str]] = None,
    ):
        """Log a tool call (REST, MCP, or Python) (fire-and-forget)."""
        if not self.agent_service or not self.run_context.run_id:
            return
        self._create_background_task(
            self._log_tool_call_impl(tool_uri, arguments, output, duration, errors)
        )

    async def _log_rag_query_impl(
        self,
        task_name: str,
        query_text: str,
        top_k: int,
        collection_name: Optional[str],
        source_ids: Optional[list[str]],
        keep_best: bool,
        output: Any,
        duration: float,
        errors: Optional[list[str]] = None,
    ):
        """Internal implementation for logging RAG query."""
        await self.agent_service.add_task(
            agent_id=self.run_context.agent_id,
            session_id=self.run_context.session_id,
            run_id=self.run_context.run_id,
            name=task_name,
            inputs={
                "text": query_text,
                "top_k": top_k,
                "collection_name": collection_name,
                "source_ids": source_ids,
                "keep_best": keep_best,
            },
            output=output,
            errors=errors,
            duration_seconds=duration,
        )

    def log_rag_query(
        self,
        task_name: str,
        query_text: str,
        top_k: int,
        collection_name: Optional[str],
        source_ids: Optional[list[str]],
        keep_best: bool,
        output: Any,
        duration: float,
        errors: Optional[list[str]] = None,
    ):
        """Log a RAG query task (fire-and-forget)."""
        if not self.agent_service or not self.run_context.run_id:
            return
        self._create_background_task(
            self._log_rag_query_impl(
                task_name, query_text, top_k, collection_name, source_ids, keep_best, output, duration, errors
            )
        )
