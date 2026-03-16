import pytest
from pydantic import BaseModel
from kavalai.agents.workflow import Workflow
from kavalai.agents.workflow_model import (
    WorkflowModel,
    LLMTask,
    TemplateModel,
)
from unittest.mock import MagicMock, AsyncMock


@pytest.mark.asyncio
async def test_workflow_templates_and_context_rendering(monkeypatch):
    # 1. Define models
    class InputModel(BaseModel):
        user_message: str
        extra: str

    class OutputModel(BaseModel):
        agent_response: str

    # 2. Mock LLMClient.chat_completions
    mock_chat_completions = AsyncMock(
        return_value=(OutputModel(agent_response="OK"), MagicMock())
    )
    monkeypatch.setattr(
        "kavalai.agents.workflow.LLMClient.chat_completions", mock_chat_completions
    )

    # 3. Create Workflow with templates
    workflow_model = WorkflowModel(
        name="test_templates",
        data_types={
            "input": {
                "type": "object",
                "properties": {
                    "user_message": {"type": "string"},
                    "extra": {"type": "string"},
                },
            },
            "output": {
                "type": "object",
                "properties": {"agent_response": {"type": "string"}},
            },
        },
        templates=[
            TemplateModel(name="examples", value="These are some examples."),
            TemplateModel(name="system_prompt", value="You are a helpful assistant."),
        ],
        tasks=[
            LLMTask(
                name="task1",
                prompt="""{{ templates.system_prompt }}
Examples:
{{ templates.examples }}

User said: {{ context.input.user_message }}
Extra: {{ context.input.extra }}""",
                inputs={
                    "user_message": {"type": "context", "value": "input.user_message"}
                },
                output="output",
            )
        ],
    )

    workflow = Workflow(workflow_model)

    # 4. Run Workflow
    input_data = {"user_message": "Hello", "extra": "Special context"}
    await workflow.run(input_data)

    # 5. Verify rendered prompt
    assert mock_chat_completions.called
    messages = mock_chat_completions.call_args[1]["messages"]
    system_content = messages[0]["content"]

    expected_prompt = """You are a helpful assistant.
Examples:
These are some examples.

User said: Hello
Extra: Special context"""

    # The make_prompt function appends "INPUT DATA:\nuser_message:Hello"
    assert expected_prompt in system_content
    assert "INPUT DATA:" in system_content
    assert "user_message:Hello" in system_content


@pytest.mark.asyncio
async def test_run_context_render_prompt_history():
    from kavalai.agents.run_context import RunContext
    from uuid import uuid4

    mock_agent_service = MagicMock()
    mock_agent_service.get_history_value = AsyncMock(
        return_value={"agent_response": "Mocked Previous result"}
    )

    rc = RunContext(
        agent_service=mock_agent_service,
        session_id=uuid4(),
        data={"input": {"user_message": "Hello"}},
        templates={"examples": "Some examples"},
    )

    prompt = "History: {{ history.task0.agent_response }}, Context: {{ context.input.user_message }}, Template: {{ templates.examples }}"
    rendered = await rc.render_prompt(prompt)

    # history.task0 returns the dict, so it's JSON serialized
    assert 'History: {"agent_response": "Mocked Previous result"}' in rendered
    assert "Context: Hello" in rendered
    assert "Template: Some examples" in rendered
