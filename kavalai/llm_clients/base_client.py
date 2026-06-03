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

import asyncio
from typing import Optional, Type

from pydantic import BaseModel

from kavalai.llm_clients.streamer import Streamer
from kavalai.llm_clients.with_retry import with_retry


class LlmClientParameters(BaseModel):
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 0.2
    reasoning_effort: Optional[str] = None
    service_tier: Optional[str] = None
    timeout_seconds: Optional[float] = 30.0


class ChatMessage(BaseModel):
    """Standard chat completion message."""

    role: Optional[str] = None
    type: Optional[str] = None
    content: Optional[str] = None


class ChatHistory(BaseModel):
    messages: list[ChatMessage]


class BaseLlmClient:
    def __init__(self, llm_client_parameters: Optional[LlmClientParameters] = None):
        if not llm_client_parameters:
            llm_client_parameters = LlmClientParameters()
        self.parameters = llm_client_parameters
        self.streamer = None

    async def stream_chat_completions(
        self,
        *,
        chat_history: ChatHistory,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> Streamer:
        """
        Execute a chat completion and return a Streamer.

        Args:
            chat_history: The history of messages.
            response_model: Optional Pydantic model for structured output.

        Returns:
            A Streamer instance that will yield the completion events.
        """
        timeout = 30.0
        if self.parameters and self.parameters.timeout_seconds:
            timeout = self.parameters.timeout_seconds

        streamer = Streamer(timeout_seconds=timeout)

        # Start the completion process in the background with retry
        asyncio.create_task(
            with_retry(
                self._run_chat_completions,
                chat_history=chat_history,
                response_model=response_model,
                streamer=streamer,
            )
        )

        return streamer

    async def chat_completions(
        self,
        *,
        chat_history: ChatHistory,
        response_model: Optional[Type[BaseModel]] = None,
    ):
        streamer = await self.stream_chat_completions(
            chat_history=chat_history, response_model=response_model
        )
        async for chunk in streamer:
            if chunk.type == "complete":
                if response_model:
                    return response_model.model_validate_json(chunk.value)
                return chunk.value
        return None

    async def prompt(
        self, system_message: str, response_model: Optional[Type[BaseModel]] = None
    ):
        history = ChatHistory(
            messages=[ChatMessage(role="system", content=system_message)]
        )
        return await self.chat_completions(
            chat_history=history, response_model=response_model
        )

    async def _run_chat_completions(
        self,
        chat_history: ChatHistory,
        response_model: Optional[Type[BaseModel]],
        streamer: Streamer,
    ):
        """
        Background task to handle the actual LLM API call and stream results.
        This method must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _run_chat_completions")


class LlmClientException(RuntimeError):
    pass


class BaseEmbeddingClient:
    pass
