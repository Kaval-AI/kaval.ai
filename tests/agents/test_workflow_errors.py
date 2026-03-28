import pytest
from kavalai.agents.workflow import Workflow, WorkflowException


@pytest.mark.asyncio
async def test_workflow_missing_template_error():
    yaml_content = """
name: test_missing_template
data_types:
  input: { type: object, properties: { msg: { type: string } } }
  output: { type: object, properties: { res: { type: string } } }
tasks:
  - name: task1
    type: llm
    prompt: "Hello {{ templates.missing_one }}"
    output: output
"""
    workflow = Workflow.from_yaml(yaml_content)

    with pytest.raises(WorkflowException) as excinfo:
        await workflow.run({"msg": "hi"})

    error_msg = str(excinfo.value)
    assert "templates.missing_one" in error_msg
    assert "Line 7" in error_msg or "line 7" in error_msg.lower()
    assert "-->    7 |   - name: task1" in error_msg


@pytest.mark.asyncio
async def test_workflow_invalid_context_path_error():
    yaml_content = """
name: test_invalid_context
data_types:
  input: { type: object, properties: { msg: { type: string } } }
  output: { type: object, properties: { res: { type: string } } }
tasks:
  - name: task1
    type: llm
    prompt: "Hello {{ context.input.wrong_field }}"
    output: output
"""
    workflow = Workflow.from_yaml(yaml_content)

    with pytest.raises(WorkflowException) as excinfo:
        await workflow.run({"msg": "hi"})

    error_msg = str(excinfo.value)
    assert "context.input.wrong_field" in error_msg
    assert "Line 7" in error_msg or "line 7" in error_msg.lower()


@pytest.mark.asyncio
async def test_workflow_invalid_history_path_error():
    yaml_content = """
name: test_invalid_history
data_types:
  input: { type: object, properties: { msg: { type: string } } }
  output: { type: object, properties: { res: { type: string } } }
tasks:
  - name: task1
    type: llm
    prompt: "Hello {{ history.task0.res }}"
    output: output
"""
    workflow = Workflow.from_yaml(yaml_content)

    # We need to simulate that task0 does not exist in history
    # For now, it might just return None and fail if we enforce it.
    with pytest.raises(WorkflowException) as excinfo:
        await workflow.run({"msg": "hi"})

    error_msg = str(excinfo.value)
    assert "history.task0.res" in error_msg
    assert "Line 7" in error_msg


@pytest.mark.asyncio
async def test_workflow_run_with_non_existent_session_id(agents_session_maker):
    from kavalai.agents.agent_service import AgentService
    from uuid import uuid4

    service = AgentService(agents_session_maker)

    yaml_content = """
name: Test Agent
description: A test agent
data_types:
  input:
    type: object
    properties:
      user_message: { type: string }
  output:
    type: object
    properties:
      reply: { type: string }
tasks:
  - name: reply_task
    type: combine
    inputs:
      reply: { type: literal, value: "hello" }
    output: output
"""
    wf = Workflow.from_yaml(yaml_content)
    wf.agent_service = service

    non_existent_session_id = uuid4()

    with pytest.raises(WorkflowException) as excinfo:
        await wf.run(
            input_data={"user_message": "hi"}, session_id=non_existent_session_id
        )

    assert f"Session with ID {non_existent_session_id} not found" in str(excinfo.value)
