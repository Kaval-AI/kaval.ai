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

import os
from typing import Optional, Type

from openai import AsyncOpenAI
from openai.types.responses import (
    ResponseTextDeltaEvent,
    ResponseRefusalDeltaEvent,
    ResponseErrorEvent,
)
from pydantic import BaseModel

from kavalai.llm_clients.base_client import (
    BaseLlmClient,
    ChatHistory,
    LlmClientParameters,
)
from kavalai.llm_clients.streamer import Streamer


class OpenAIClient(BaseLlmClient):
    """
    OpenAI LLM client implementation using the Responses API and Streamer.
    """

    def __init__(
        self,
        model: str,
        llm_client_parameters: Optional[LlmClientParameters] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the OpenAI client.

        Args:
            model: The OpenAI model name (e.g., 'gpt-4o').
            llm_client_parameters: Optional parameters like temperature, top_p, etc.
            api_key: Optional API key (falls back to OPENAI_API_KEY env var).
            base_url: Optional base URL for the API.
        """
        super().__init__(llm_client_parameters)
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url

        timeout = 30.0
        if self.parameters and self.parameters.timeout_seconds:
            timeout = self.parameters.timeout_seconds

        self.client = AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=timeout
        )

    async def _run_chat_completions(
        self,
        chat_history: ChatHistory,
        response_model: Optional[Type[BaseModel]],
        streamer: Streamer,
    ):
        """
        Background task to handle the actual OpenAI API call and stream results.
        """
        value_streamer = streamer.get_value_streamer(
            "response", response_model=response_model
        )

        messages = []
        for msg in chat_history.messages:
            message_dict = {"role": msg.role, "content": msg.content}
            messages.append(message_dict)

        call_kwargs = {
            "model": self.model,
            "input": messages,
        }

        if self.parameters:
            if self.parameters.temperature is not None:
                call_kwargs["temperature"] = self.parameters.temperature
            if self.parameters.top_p is not None:
                call_kwargs["top_p"] = self.parameters.top_p
            if self.parameters.service_tier is not None:
                call_kwargs["service_tier"] = self.parameters.service_tier
            if self.parameters.reasoning_effort is not None:
                call_kwargs["reasoning_effort"] = self.parameters.reasoning_effort

        if response_model:
            call_kwargs["text_format"] = response_model

        async with self.client.responses.stream(**call_kwargs) as stream:
            async for event in stream:
                if isinstance(event, ResponseTextDeltaEvent):
                    await value_streamer.stream_partial(event.delta)
                elif isinstance(event, ResponseRefusalDeltaEvent):
                    await value_streamer.stream_partial(event.delta)
                elif isinstance(event, ResponseErrorEvent):
                    # We raise here to let the background task fail
                    raise RuntimeError(f"OpenAI Stream Error: {event.error}")

        await value_streamer.stream_complete()
