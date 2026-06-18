from kavalai.workflow.state import WorkflowState


def test_defaults():
    state = WorkflowState(workflow_name="wf")
    assert state.status == "pending"
    assert state.current_node is None
    assert state.trace == []
    assert state.data == {}
    assert state.output_data is None
    assert state.invocation_id is None
    assert state.token_usage is None


def test_json_roundtrip():
    state = WorkflowState(
        workflow_name="wf",
        status="completed",
        current_node="end",
        trace=["start", "mid", "end"],
        data={"input": {"user_message": "hi"}, "count": 3},
        input_data={"user_message": "hi"},
        output_data={"agent_response": "done"},
        invocation_id="abcd1234",
        token_usage={"model_calls": 1, "total_tokens": 5},
        run_id="run-1",
        session_id="sess-1",
        agent_id="agent-1",
    )
    restored = WorkflowState.from_json(state.to_json())
    assert restored == state
    assert restored.trace == ["start", "mid", "end"]
    assert restored.output_data == {"agent_response": "done"}
    assert restored.invocation_id == "abcd1234"
    assert restored.token_usage == {"model_calls": 1, "total_tokens": 5}


def test_to_json_is_string():
    state = WorkflowState(workflow_name="wf")
    payload = state.to_json()
    assert isinstance(payload, str)
    assert '"workflow_name":"wf"' in payload
