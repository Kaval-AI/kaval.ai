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
import os
from typing import Optional, Type

import ollama
from pydantic import BaseModel

from kavalai.llm_clients.base_client import (
    BaseLlmClient,
    ChatHistory,
    LlmClientParameters,
)
from kavalai.llm_clients.streamer import Streamer


class OllamaClient(BaseLlmClient):
    """
    Ollama LLM client implementation using the Streamer.
    """

    def __init__(
        self,
        model: str,
        llm_client_parameters: Optional[LlmClientParameters] = None,
        host: Optional[str] = None,
    ):
        """
        Initialize the Ollama client.

        Args:
            model: The Ollama model name (e.g., 'llama3').
            llm_client_parameters: Optional parameters like temperature, top_p, etc.
            host: Optional Ollama host (falls back to OLLAMA_HOST env var).
        """
        super().__init__(llm_client_parameters)
        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

        self.timeout = 30.0
        if self.parameters and self.parameters.timeout_seconds:
            self.timeout = self.parameters.timeout_seconds

        self.client = ollama.AsyncClient(host=self.host, timeout=self.timeout)

    async def chat_completions(
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
        streamer = Streamer(timeout_seconds=self.timeout)

        # Start the completion process in the background
        asyncio.create_task(
            self._run_chat_completions(chat_history, response_model, streamer)
        )

        return streamer

    async def _run_chat_completions(
        self,
        chat_history: ChatHistory,
        response_model: Optional[Type[BaseModel]],
        streamer: Streamer,
    ):
        """
        Background task to handle the actual Ollama API call and stream results.
        """
        value_streamer = streamer.get_value_streamer(
            "response", response_model=response_model
        )

        messages = []
        for msg in chat_history.messages:
            message_dict = {"role": msg.role, "content": msg.content}
            messages.append(message_dict)

        options = {}
        if self.parameters:
            if self.parameters.temperature is not None:
                options["temperature"] = self.parameters.temperature
            if self.parameters.top_p is not None:
                options["top_p"] = self.parameters.top_p

        call_kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": options,
        }

        if response_model:
            # Ollama supports 'format': 'json'
            call_kwargs["format"] = "json"

        try:
            async for chunk in await self.client.chat(**call_kwargs):
                if "message" in chunk and "content" in chunk["message"]:
                    delta = chunk["message"]["content"]
                    await value_streamer.stream_partial(delta)

            await value_streamer.stream_complete()
        except Exception as e:
            # We follow the OpenAIClient pattern and let the task fail
            raise e
