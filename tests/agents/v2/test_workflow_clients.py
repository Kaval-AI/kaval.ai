import pytest

from kavalai.agents.v2.workflow import clients
from kavalai.llm_clients.base_client import LlmClientParameters
from kavalai.llm_clients.v2.gemini_client import GeminiClient
from kavalai.llm_clients.v2.ollama_client import OllamaClient
from kavalai.llm_clients.v2.openai_client import OpenAIClient


def test_build_parameters_filters_unknown():
    params = clients.build_parameters(
        {"temperature": 0.3, "top_p": 0.9, "unknown_key": "ignored"}
    )
    assert isinstance(params, LlmClientParameters)
    assert params.temperature == 0.3
    assert params.top_p == 0.9


def test_build_parameters_none():
    params = clients.build_parameters(None)
    assert isinstance(params, LlmClientParameters)


def test_make_client_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = clients.make_client("openai/gpt-4o")
    assert isinstance(client, OpenAIClient)
    assert client.model == "gpt-4o"


def test_make_client_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = clients.make_client("gemini/gemini-2.0", clients.build_parameters({}))
    assert isinstance(client, GeminiClient)
    assert client.model == "gemini-2.0"


def test_make_client_ollama():
    client = clients.make_client("ollama/llama3")
    assert isinstance(client, OllamaClient)
    assert client.model == "llama3"


def test_make_client_requires_provider():
    with pytest.raises(ValueError, match="provider/model"):
        clients.make_client("gpt-4o")


def test_make_client_unsupported_provider():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        clients.make_client("anthropic/claude")
