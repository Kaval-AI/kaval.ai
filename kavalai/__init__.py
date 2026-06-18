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

# Kaval.AI public API.
#
# The names below are the supported, stable import surface. Everything a user
# typically needs is reachable as ``from kavalai import X``. The headline
# entry points are:
#
#   * ``WorkflowEngine`` / ``WorkflowBuilder`` -- define and run workflows.
#   * ``Agent``                                -- the multi-step tool-calling agent.
#   * ``FunctionKernel`` / ``pythontool``      -- register and call tools.
#   * ``OpenAIClient`` / ``GeminiClient`` / ``OllamaClient`` -- LLM backends.
#   * ``RagService``                           -- index and query embeddings.
#
# The persistence-layer ORM table classes (Agent row, Run, Task, ...) live in
# ``kavalai.agents.db`` to keep the runtime names (``Agent`` the agent,
# ``ModelCallStat`` the stats model) unambiguous at the top level.

# --- Workflow engine -------------------------------------------------------
from kavalai.workflow import (
    WorkflowEngine,
    WorkflowBuilder,
    WorkflowState,
    WorkflowGraph,
    Node,
    StartNode,
    EndNode,
    LLMNode,
    AgentNode,
    FunctionNode,
    IfNode,
    SwitchNode,
    evaluate_expression,
    evaluate_bool,
    evaluate_value,
    ExpressionError,
    DataStorage,
    RunHandle,
    ChatMsg,
    SqliteDataStorage,
    TaskLogger,
    StatsBridge,
    TokenAccumulator,
    SqliteTaskLogger,
)
from kavalai.workflow.clients import make_client

# --- Agent & tools ---------------------------------------------------------
from kavalai.agents.agent import Agent, ToolCall
from kavalai.agents.run_context import RunContext
from kavalai.agents.workflow_model import (
    RestServer,
    McpServer,
    PythonFunction,
    TemplateModel,
    ArgumentInfo,
    WorkflowException,
)
from kavalai.functionkernel import (
    FunctionKernel,
    FunctionKernelException,
    pythontool,
)

# --- LLM & embedding clients ----------------------------------------------
from kavalai.llm_clients.base_client import (
    BaseLlmClient,
    LlmClientParameters,
    ChatHistory,
    ChatMessage,
    ModelCallStat,
    ModelStatsReceiver,
    ModelStatsLogger,
)
from kavalai.llm_clients.openai_client import OpenAIClient
from kavalai.llm_clients.gemini_client import GeminiClient
from kavalai.llm_clients.ollama_client import OllamaClient
from kavalai.llm_clients.embeddings import make_embedding_client

# --- Streaming -------------------------------------------------------------
from kavalai.llm_clients.streamer import (
    Streamer,
    StreamContent,
    ValueStreamer,
    StreamerTimeoutException,
)

# --- RAG, normalization, persistence --------------------------------------
from kavalai.agents.rag_service import RagService
from kavalai.normalizer import Normalizer
from kavalai.agents.db import db_manager

__all__ = [
    # Workflow engine
    "WorkflowEngine",
    "WorkflowBuilder",
    "WorkflowState",
    "WorkflowGraph",
    "Node",
    "StartNode",
    "EndNode",
    "LLMNode",
    "AgentNode",
    "FunctionNode",
    "IfNode",
    "SwitchNode",
    "evaluate_expression",
    "evaluate_bool",
    "evaluate_value",
    "ExpressionError",
    "DataStorage",
    "RunHandle",
    "ChatMsg",
    "SqliteDataStorage",
    "TaskLogger",
    "StatsBridge",
    "TokenAccumulator",
    "SqliteTaskLogger",
    "make_client",
    # Agent & tools
    "Agent",
    "ToolCall",
    "RunContext",
    "RestServer",
    "McpServer",
    "PythonFunction",
    "TemplateModel",
    "ArgumentInfo",
    "WorkflowException",
    "FunctionKernel",
    "FunctionKernelException",
    "pythontool",
    # LLM & embedding clients
    "BaseLlmClient",
    "LlmClientParameters",
    "ChatHistory",
    "ChatMessage",
    "ModelCallStat",
    "ModelStatsReceiver",
    "ModelStatsLogger",
    "OpenAIClient",
    "GeminiClient",
    "OllamaClient",
    "make_embedding_client",
    # Streaming
    "Streamer",
    "StreamContent",
    "ValueStreamer",
    "StreamerTimeoutException",
    # RAG, normalization, persistence
    "RagService",
    "Normalizer",
    "db_manager",
]
