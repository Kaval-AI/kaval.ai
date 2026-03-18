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
from loguru import logger

from kavalai.agents.agent_service import AgentService
from kavalai.agents.run_context import RunContext


class TaskLogger:
    """Component for logging tasks to the database with consistent structure.

    All methods handle errors internally and log them without raising exceptions.
    """

    def __init__(self, agent_service: Optional[AgentService], run_context: RunContext):
        self.agent_service = agent_service
        self.run_context = run_context

    async def log_llm_task(
        self,
        task_name: str,
        prompt: str,
        input_data: dict,
        output: Any,
        duration: float,
        errors: Optional[list[str]] = None,
    ):
        """Log an LLM task with full prompt."""
        if not self.agent_service or not self.run_context.run_id:
            return

        try:
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
        except Exception as e:
            logger.error(f"Failed to log LLM task '{task_name}': {e}")

    async def log_agent_task(
        self,
        task_name: str,
        system_prompt: str,
        input_data: dict,
        output: Any,
        duration: float,
        errors: Optional[list[str]] = None,
    ):
        """Log an agent task with full system prompt."""
        if not self.agent_service or not self.run_context.run_id:
            return

        try:
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
        except Exception as e:
            logger.error(f"Failed to log agent task '{task_name}': {e}")

    async def log_tool_call(
        self,
        tool_uri: str,
        arguments: dict,
        output: Any,
        duration: float,
        errors: Optional[list[str]] = None,
    ):
        """Log a tool call (REST, MCP, or Python)."""
        if not self.agent_service or not self.run_context.run_id:
            return

        try:
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
        except Exception as e:
            logger.error(f"Failed to log tool call '{tool_uri}': {e}")

    async def log_rag_query(
        self,
        task_name: str,
        query_text: str,
        top_k: int,
        collection_name: Optional[str],
        output: Any,
        duration: float,
        errors: Optional[list[str]] = None,
    ):
        """Log a RAG query task."""
        if not self.agent_service or not self.run_context.run_id:
            return

        try:
            await self.agent_service.add_task(
                agent_id=self.run_context.agent_id,
                session_id=self.run_context.session_id,
                run_id=self.run_context.run_id,
                name=task_name,
                inputs={
                    "text": query_text,
                    "top_k": top_k,
                    "collection_name": collection_name,
                },
                output=output,
                errors=errors,
                duration_seconds=duration,
            )
        except Exception as e:
            logger.error(f"Failed to log RAG query '{task_name}': {e}")
