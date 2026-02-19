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
import asyncio
import logging
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field

from kavalai.agents.workflow_model import (
    WorkflowException,
    TypeInputInfo,
    AgentTask,
    RestTask,
    McpTask,
)
from kavalai.llm_clients.llm_client import chat_completions
from kavalai.llm_clients.common import Streamer

logger = logging.getLogger(__name__)


def _make_prompt(prompt: str | None, input_data: dict) -> str:
    if not prompt:
        prompt = "You are a planning agent. Decide next action."
    pieces = [prompt]
    if len(input_data) > 0:
        pieces.append("INPUT DATA:")
        for key, value in input_data.items():
            if isinstance(value, BaseModel):
                value = value.model_dump_json()
            pieces.append(f"{key}:{value}")
    return "\n".join(pieces)


class _ToolDirective(BaseModel):
    action: str  # "tool" | "finish"
    tool_kind: Optional[str] = None  # "rest" | "mcp"
    server: Optional[str] = None
    name: Optional[str] = None
    arguments: Dict[str, Any] = Field(default_factory=dict)
    result_key: Optional[str] = None
    final_output_key: Optional[str] = None


class PlanningAgent:
    def __init__(self, workflow: Any):
        # Keep a reference to Workflow to reuse tool runners and context
        self.workflow = workflow

    async def run(self, task: AgentTask, run_context, queue: asyncio.Queue | None):
        # 1) Resolve initial inputs
        input_data = {}
        for name, info in task.inputs.items():
            if info.value is None and info.name is None:
                info = info.model_copy(update={"value": name})
            input_data[name] = await run_context.resolve_input_info(info)

        # 2) Build messages
        messages = [
            {"role": "system", "content": "You must respond with JSON only."},
            {"role": "user", "content": _make_prompt(task.prompt, input_data)},
        ]

        # 3) Loop
        steps = 0
        while steps < task.max_steps:
            steps += 1

            async def _ask():
                return await chat_completions(
                    model=self.workflow.workflow_model.llm_model
                    or os.environ["KAVALAI_DEFAULT_LLM_MODEL"],
                    response_model=_ToolDirective,
                    messages=messages,
                )

            if getattr(task, "timeout", None):
                try:
                    directive, stats = await asyncio.wait_for(
                        _ask(), timeout=task.timeout
                    )
                except asyncio.TimeoutError:
                    raise WorkflowException(
                        f"Special agent step timed out after {task.timeout}s"
                    )
            else:
                directive, stats = await _ask()

            # Record the LLM call as a task with directive + stats
            if self.workflow.agent_service and run_context.run_id:
                await self.workflow.agent_service.add_task(
                    agent_id=run_context.agent_id,
                    session_id=run_context.session_id,
                    run_id=run_context.run_id,
                    inputs={"messages": messages},
                    output={
                        "directive": directive.model_dump(),
                        "stats": stats.model_dump()
                        if hasattr(stats, "model_dump")
                        else None,
                    },
                )

            if directive.action == "finish":
                final_key = directive.final_output_key or (
                    task.output if isinstance(task.output, str) else None
                )
                if not final_key:
                    break
                value = run_context.resolve_context_value(final_key)
                if value is None and isinstance(task.output, str):
                    value = input_data.get(final_key)
                if isinstance(task.output, str) and value is not None:
                    run_context.data[task.output] = value
                    if task.stream and queue is not None:
                        streamer = Streamer(task.output, queue)
                        await streamer.stream_complete(
                            value if isinstance(value, str) else str(value)
                        )
                break

            if directive.action == "tool":
                if directive.tool_kind == "mcp":
                    if (
                        task.allowed_mcp_servers
                        and directive.server not in task.allowed_mcp_servers
                    ):
                        raise WorkflowException(
                            f"MCP server '{directive.server}' not allowed for task '{task.name}'."
                        )

                if directive.tool_kind == "rest":
                    proxy = RestTask(
                        name=f"{task.name}__step{steps}__tool",
                        inputs={
                            k: TypeInputInfo(type="literal", value=v)
                            for k, v in (directive.arguments or {}).items()
                        },
                        output=directive.result_key or f"__tool_result_{steps}",
                        tool=directive.name,
                        rest_server=directive.server,
                    )
                    await self.workflow.run_rest_tool(proxy, run_context, queue)
                elif directive.tool_kind == "mcp":
                    proxy = McpTask(
                        name=f"{task.name}__step{steps}__tool",
                        inputs={
                            k: TypeInputInfo(type="literal", value=v)
                            for k, v in (directive.arguments or {}).items()
                        },
                        output=directive.result_key or f"__tool_result_{steps}",
                        tool=directive.name,
                        mcp_server=directive.server,
                    )
                    await self.workflow.run_mcp_tool(proxy, run_context, queue)
                else:
                    raise WorkflowException(f"Unknown tool_kind: {directive.tool_kind}")

                # Feed tool result back
                tool_val = run_context.resolve_context_value(proxy.output)
                messages.append(
                    {
                        "role": "system",
                        "content": f"TOOL_RESULT {proxy.output}: {tool_val}",
                    }
                )
                continue

            raise WorkflowException("Invalid directive from LLM in special agent.")

        if isinstance(task.output, str) and task.output not in run_context.data:
            run_context.data[task.output] = run_context.data.get(task.output, None)
