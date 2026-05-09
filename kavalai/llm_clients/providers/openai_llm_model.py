import io
import json
import os
import time
from typing import Any, Dict, List, Optional, Type, Tuple

from kavalai.llm_clients.base_client import BaseLlmModel
from openai import AsyncOpenAI
from openai.types.responses import (
    ResponseTextDeltaEvent,
    ResponseRefusalDeltaEvent,
    ResponseErrorEvent,
    ResponseCompletedEvent,
)
from pydantic import BaseModel

from kavalai.agents.db import ModelCallStat
from kavalai.llm_clients.common import (
    create_model_call_stat,
    safe_parse_json,
    Streamer,
)


class OpenAILlmModelConfig(BaseModel):
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout_seconds: float = 30.0
    reasoning_effort: str = "low"
    service_tier: Optional[str] = None


class OpenAILlmModel(BaseLlmModel):
    def __init__(self, config: OpenAILlmModelConfig):
        super().__init__()
        self.config = config
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    async def chat_completions(
        self,
        messages: List[Dict[str, Any]],
        response_model: Optional[Type[BaseModel]] = None,
        streamer: Optional[Streamer] = None,
        stream_delta: bool = False,
        **kwargs,
    ) -> Tuple[Any, ModelCallStat]:
        start_time = time.perf_counter()

        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                new_content = []
                for item in content:
                    if item.get("type") == "text":
                        new_content.append(
                            {"type": "input_text", "text": item.get("text")}
                        )
                    elif item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        new_content.append({"type": "input_image", "image_url": url})
                    else:
                        new_content.append(item)
                msg["content"] = new_content

        # Filter out unsupported parameters for the Responses API
        call_kwargs = {
            "model": self.config.model_name,
            "input": messages,
            **{k: v for k, v in kwargs.items() if v is not None},
        }

        if "reasoning_effort" not in call_kwargs and self.config.reasoning_effort:
            call_kwargs["reasoning_effort"] = self.config.reasoning_effort

        if "service_tier" not in call_kwargs and self.config.service_tier:
            call_kwargs["service_tier"] = self.config.service_tier

        if response_model and issubclass(response_model, BaseModel):
            call_kwargs["text_format"] = response_model
        elif response_model:
            raise ValueError("response_model must be a pydantic BaseModel")

        buffer = io.StringIO()
        input_tokens = 0
        output_tokens = 0
        async with self.client.responses.stream(**call_kwargs) as stream:
            async for event in stream:
                if isinstance(event, ResponseTextDeltaEvent):
                    buffer.write(event.delta)
                    if streamer is not None:
                        if stream_delta:
                            await streamer.stream_partial(event.delta)
                        else:
                            value = (
                                safe_parse_json(buffer.getvalue())
                                if response_model
                                else buffer.getvalue()
                            )
                            if isinstance(value, (dict, list)):
                                value = json.dumps(value)
                            await streamer.stream_partial(value)
                elif isinstance(event, ResponseRefusalDeltaEvent):
                    buffer.write(event.delta)
                    if streamer is not None:
                        if stream_delta:
                            await streamer.stream_partial(event.delta)
                        else:
                            value = (
                                safe_parse_json(buffer.getvalue())
                                if response_model
                                else buffer.getvalue()
                            )
                            if isinstance(value, (dict, list)):
                                value = json.dumps(value)
                            await streamer.stream_partial(value)
                elif isinstance(event, ResponseErrorEvent):
                    raise RuntimeError(event.error)
                elif isinstance(event, ResponseCompletedEvent):
                    usage = event.response.usage
                    input_tokens = usage.input_tokens
                    output_tokens = usage.output_tokens

        # Stream the final complete value.
        if streamer is not None:
            value = (
                safe_parse_json(buffer.getvalue())
                if response_model
                else buffer.getvalue()
            )
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            await streamer.stream_complete(value)

        result = buffer.getvalue()
        if response_model:
            result = response_model.model_validate(safe_parse_json(result))

        duration = time.perf_counter() - start_time

        stats = create_model_call_stat(
            call_type="llm",
            model=f"openai/{self.config.model_name}",
            duration_sections=duration,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            cost=None,
            response_data=result.model_dump()
            if hasattr(result, "model_dump")
            else result,
        )
        stats.currency = None
        return result, stats
