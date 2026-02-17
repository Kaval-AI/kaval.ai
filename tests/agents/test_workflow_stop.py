import pytest
from kavalai.agents.workflow import Workflow


@pytest.mark.asyncio
async def test_workflow_stop_field():
    workflow_yaml = """
name: StopTest
data_types:
  input:
    type: object
    properties:
      should_stop:
        type: boolean
  output:
    type: object
    properties:
      task1_run:
        type: boolean
        required: false
        default: false
      task2_run:
        type: boolean
        required: false
        default: false
tasks:
  - name: Task 1
    stop: true
    output:
      task1_run:
        type: literal
        value: true
  - name: Task 2
    output:
      task2_run:
        type: literal
        value: true
"""
    workflow = Workflow.from_yaml(workflow_yaml)

    # Task 1 has stop: true, so Task 2 should NOT run
    result = await workflow.run({"should_stop": True})

    assert result.data is not None
    assert result.data.task1_run is True
    assert result.data.task2_run is False


@pytest.mark.asyncio
async def test_workflow_stop_with_when():
    workflow_yaml = """
name: StopWhenTest
data_types:
  input:
    type: object
    properties:
      stop_early:
        type: boolean
  output:
    type: object
    properties:
      step:
        type: string
tasks:
  - name: Step 1
    when:
      eq: [ { type: context, value: input.stop_early }, true ]
    stop: true
    output:
      step:
        type: literal
        value: "stopped at step 1"
  - name: Step 2
    output:
      step:
        type: literal
        value: "reached step 2"
"""
    workflow = Workflow.from_yaml(workflow_yaml)

    # Case 1: stop_early is true. Step 1 runs and stops the workflow.
    result1 = await workflow.run({"stop_early": True})
    assert result1.data.step == "stopped at step 1"

    # Case 2: stop_early is false. Step 1 is skipped, so stop: true is not triggered.
    # Step 2 should run.
    result2 = await workflow.run({"stop_early": False})
    assert result2.data.step == "reached step 2"
