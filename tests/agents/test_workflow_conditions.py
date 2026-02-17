import pytest
from kavalai.agents.workflow import Workflow


@pytest.mark.asyncio
async def test_workflow_conditional_tasks():
    workflow_yaml = """
name: ConditionalTest
data_types:
  input:
    type: object
    properties:
      run_task:
        type: boolean
      value:
        type: integer
  output:
    type: object
    properties:
      result:
        type: string
tasks:
  - name: Conditional Task
    when:
      eq: [ { type: context, value: input.run_task }, true ]
    inputs:
      val:
        type: context
        value: input.value
    output:
      result:
        type: literal
        value: "Task executed"
  - name: GT Task
    when:
      gt: [ { type: context, value: input.value }, 10 ]
    output:
      result:
        type: literal
        value: "Value is large"
  - name: All Task
    when:
      all:
        - eq: [ { type: context, value: input.run_task }, true ]
        - gt: [ { type: context, value: input.value }, 5 ]
    output:
      result:
        type: literal
        value: "All conditions met"
"""
    workflow = Workflow.from_yaml(workflow_yaml)

    # Case 1: run_task=False, value=5 -> No tasks should run, output should be empty/None
    result1 = await workflow.run({"run_task": False, "value": 5})
    assert result1.data is None

    # Case 2: run_task=True, value=5 -> Only "Conditional Task" should run
    result2 = await workflow.run({"run_task": True, "value": 5})
    assert result2.data.result == "Task executed"

    # Case 3: run_task=False, value=15 -> Only "GT Task" should run
    result3 = await workflow.run({"run_task": False, "value": 15})
    assert result3.data.result == "Value is large"

    # Case 4: run_task=True, value=15 -> All tasks run, last one wins for "result"
    result4 = await workflow.run({"run_task": True, "value": 15})
    assert result4.data.result == "All conditions met"


@pytest.mark.asyncio
async def test_workflow_conditional_any_not():
    workflow_yaml = """
name: AnyNotTest
data_types:
  input:
    type: object
    properties:
      val:
        type: integer
  output:
    type: object
    properties:
      result:
        type: string
tasks:
  - name: Any Task
    when:
      any:
        - eq: [ { type: context, value: input.val }, 1 ]
        - eq: [ { type: context, value: input.val }, 2 ]
    output:
      result:
        type: literal
        value: "One or Two"
  - name: Not Task
    when:
      not:
        eq: [ { type: context, value: input.val }, 1 ]
    output:
      result:
        type: literal
        value: "Not One"
"""
    workflow = Workflow.from_yaml(workflow_yaml)

    # val=1 -> "Any Task" runs, "Not Task" does not. Result: "One or Two"
    result1 = await workflow.run({"val": 1})
    assert result1.data.result == "One or Two"

    # val=2 -> "Any Task" runs, "Not Task" runs. Result: "Not One"
    result2 = await workflow.run({"val": 2})
    assert result2.data.result == "Not One"

    # val=3 -> "Any Task" does not run, "Not Task" runs. Result: "Not One"
    result3 = await workflow.run({"val": 3})
    assert result3.data.result == "Not One"


@pytest.mark.asyncio
async def test_workflow_conditional_nested_paths():
    workflow_yaml = """
name: NestedPathTest
data_types:
  input:
    type: object
    properties:
      criteria:
        type: object
        properties:
          keywords:
            type: array
            items:
              type: string
  output:
    type: object
    properties:
      result:
        type: string
tasks:
  - name: Search
    when:
      gt: [ { type: context, value: "input.criteria.keywords.length" }, 0 ]
    output:
      result:
        type: literal
        value: "Found keywords"
"""
    workflow = Workflow.from_yaml(workflow_yaml)

    # Empty keywords
    result1 = await workflow.run({"criteria": {"keywords": []}})
    assert result1.data is None

    # Some keywords
    result2 = await workflow.run({"criteria": {"keywords": ["hotel", "paris"]}})
    assert result2.data.result == "Found keywords"


@pytest.mark.asyncio
async def test_workflow_conditional_contains():
    workflow_yaml = """
name: ContainsTest
data_types:
  input:
    type: object
    properties:
      tags:
        type: array
        items:
          type: string
  output:
    type: object
    properties:
      result:
        type: string
tasks:
  - name: Tag Check
    when:
      contains: [ { type: context, value: "input.tags" }, "vip" ]
    output:
      result:
        type: literal
        value: "VIP Customer"
"""
    workflow = Workflow.from_yaml(workflow_yaml)

    # No vip tag
    result1 = await workflow.run({"tags": ["new", "newsletter"]})
    assert result1.data is None

    # With vip tag
    result2 = await workflow.run({"tags": ["vip", "newsletter"]})
    assert result2.data.result == "VIP Customer"


@pytest.mark.asyncio
async def test_workflow_conditional_invalid_operator_length():
    workflow_yaml = """
name: InvalidLengthTest
data_types:
  input:
    type: object
    properties:
      val:
        type: integer
  output:
    type: object
    properties:
      result:
        type: string
tasks:
  - name: Invalid Task
    when:
      eq: [ { type: context, value: input.val }, 1 , 2]
    output:
      result:
        type: literal
        value: "Should fail"
"""
    # The error now happens during Workflow.from_yaml (initialization) due to Pydantic validator
    with pytest.raises(
        ValueError, match="Operator 'eq' requires a list of 2 operands."
    ):
        Workflow.from_yaml(workflow_yaml)


@pytest.mark.asyncio
async def test_workflow_conditional_operators():
    workflow_yaml = """
name: OperatorTest
data_types:
  input:
    type: object
    properties:
      val:
        type: integer
  output:
    type: object
    properties:
      result:
        type: string
tasks:
  - name: GTE Task
    when:
      gte: [ { type: context, value: input.val }, 10 ]
    output:
      result:
        type: literal
        value: "GTE 10"
  - name: LT Task
    when:
      lt: [ { type: context, value: input.val }, 5 ]
    output:
      result:
        type: literal
        value: "LT 5"
  - name: NotEQ Task
    when:
      not_eq: [ { type: context, value: input.val }, 7 ]
    output:
      result:
        type: literal
        value: "Not 7"
"""
    workflow = Workflow.from_yaml(workflow_yaml)

    # val=12 -> GTE runs, NotEQ runs. Result: "Not 7"
    result1 = await workflow.run({"val": 12})
    assert result1.data.result == "Not 7"

    # val=4 -> LT runs, NotEQ runs. Result: "Not 7"
    result2 = await workflow.run({"val": 4})
    assert result2.data.result == "Not 7"

    # val=7 -> GTE (false), LT (false), NotEQ (false). Result: None
    result3 = await workflow.run({"val": 7})
    assert result3.data is None

    # val=10 -> GTE runs, NotEQ runs. Result: "Not 7"
    result4 = await workflow.run({"val": 10})
    assert result4.data.result == "Not 7"
