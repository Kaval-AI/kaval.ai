import pytest
from kavalai.agents.workflow import Workflow
from kavalai.agents.agent_service import AgentService


@pytest.mark.asyncio
async def test_workflow_load_from_history(agents_db, monkeypatch):
    service = AgentService(agents_db)

    # 1. Setup first run to populate history
    yaml_1 = """
name: Run1
description: First run
data_types:
  input:
    type: object
    properties:
      user_message: { type: string }
  output:
    type: object
    properties:
      search_results: { type: string }
tasks:
  - name: combine
    inputs:
      search_results: { type: literal, value: "initial results" }
    output:
      search_results: { type: literal, value: "initial results" }
"""
    wf1 = Workflow.from_yaml(yaml_1)
    wf1.agent_service = service

    res1 = await wf1.run(
        input_data={"user_message": "hello"}, external_id="session-123"
    )
    session_id = res1.session_id
    assert session_id is not None

    # 2. Setup second run to load from history
    yaml_2 = """
name: Run2
description: Second run
data_types:
  input:
    type: object
    properties:
      user_message: { type: string }
  output:
    type: object
    properties:
      prev_results: { type: string }
      current_msg: { type: string }
tasks:
  - name: load_and_combine
    inputs:
      prev_results: { type: load_from_history, value: search_results }
      current_msg: { type: context, value: input.user_message }
    output:
      prev_results: { type: load_from_history, value: search_results }
      current_msg: { type: context, value: input.user_message }
"""
    wf2 = Workflow.from_yaml(yaml_2)
    wf2.agent_service = service

    res2 = await wf2.run(input_data={"user_message": "bye"}, session_id=session_id)

    assert res2.data.prev_results == "initial results"
    assert res2.data.current_msg == "bye"


@pytest.mark.asyncio
async def test_workflow_condition_with_history(agents_db, monkeypatch):
    service = AgentService(agents_db)

    # 1. Populate history
    yaml_1 = """
name: Run1
data_types:
  input: { type: object, properties: { msg: { type: string } } }
  output: { type: object, properties: { status: { type: string } } }
tasks:
  - name: t1
    inputs: { status: { type: literal, value: "completed" } }
    output: { status: { type: literal, value: "completed" } }
"""
    wf1 = Workflow.from_yaml(yaml_1)
    wf1.agent_service = service
    res1 = await wf1.run(input_data={"msg": "start"}, external_id="session-456")
    session_id = res1.session_id

    # 2. Run with condition checking history
    yaml_2 = """
name: Run2
data_types:
  input: { type: object, properties: { msg: { type: string } } }
  output: { type: object, properties: { result: { type: string } } }
tasks:
  - name: conditional_task
    when:
      eq:
        - { type: load_from_history, value: status }
        - "completed"
    inputs:
      result: { type: literal, value: "it worked" }
    output: { result: { type: literal, value: "it worked" } }
"""
    wf2 = Workflow.from_yaml(yaml_2)
    wf2.agent_service = service
    res2 = await wf2.run(input_data={"msg": "next"}, session_id=session_id)

    assert res2.data is not None
    assert res2.data.result == "it worked"

    # 3. Run with condition that should fail
    yaml_3 = """
name: Run3
data_types:
  input: { type: object, properties: { msg: { type: string } } }
  output: { type: object, properties: { result: { type: string } } }
tasks:
  - name: conditional_task
    when:
      eq:
        - { type: load_from_history, value: status }
        - "failed"
    inputs:
      result: { type: literal, value: "should not happen" }
    output: { result: { type: literal, value: "should not happen" } }
"""
    wf3 = Workflow.from_yaml(yaml_3)
    wf3.agent_service = service
    res3 = await wf3.run(input_data={"msg": "next"}, session_id=session_id)
    assert res3.data is None
