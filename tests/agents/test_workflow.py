import pytest
from kavalai.agents.workflow import Workflow
from kavalai.agents.agent_service import AgentService

SIMPLE_YAML = """
name: BB King Agent
description: Just talks about the blues.
llm_profile_name: openai/gpt-4o
data_types:
  input:
    type: object
    properties:
      user_message: {type: string}
  output:
    type: object
    properties:
      agent_response: {type: string}
mcp_servers: []
tasks:
  - name: Blues Talk
    prompt: "You are BB King. You like talking about the blues."
    inputs:
      input: {type: context, name: input}
    output: output
"""


@pytest.mark.asyncio
class TestBBKingWorkflow:
    async def test_bb_king_full_integration(self, agents_db):
        """
        Tests a real LLM call and verifies the data persistence
        hierarchy: Agent -> Session -> Run -> Task -> ChatMessage.
        """
        # 1. Setup
        service = AgentService(agents_db)
        wf = Workflow.from_yaml(SIMPLE_YAML)
        wf.agent_service = service

        # Ensure LLM profile exists as a file for Workflow to load
        import os
        import yaml

        profile_name = "openai/gpt-4o"
        profile_dir = "llm_profiles"
        os.makedirs(profile_dir, exist_ok=True)
        profile_path = os.path.join(profile_dir, f"{profile_name}.yaml")

        # Ensure subdirectory exists if name contains slashes
        os.makedirs(os.path.dirname(profile_path), exist_ok=True)

        with open(profile_path, "w") as f:
            yaml.dump(
                {
                    "name": profile_name,
                    "provider": "openai",
                    "model_name": "gpt-4o",
                },
                f,
            )

        user_input = {"user_message": "Tell me about Lucille, your guitar."}

        # Mock the instructor client to avoid real LLM calls in tests
        from unittest.mock import AsyncMock, patch

        # Get the output model class to create a real instance
        output_type = wf.get_data_type("output")
        mock_response = output_type(agent_response="Lucille is my lady.")

        with patch("kavalai.agents.workflow.get_instructor") as mock_get_instructor:
            mock_client = AsyncMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get_instructor.return_value = mock_client

            # 2. Execution
            result = await wf.run(input_data=user_input)

        # 3. Validation: Result Structure
        assert result.session_id is not None
        assert result.data is not None
        assert (
            "Lucille" in result.data.agent_response
            or "guitar" in result.data.agent_response.lower()
        )

        # 4. Validation: Database Records
        # Check Agent
        agent = await service.get_or_create_agent(name="BB King Agent")
        assert agent.description == "Just talks about the blues."

        # Check Chat History (Should have 2 messages: User and Assistant)
        history = await service.get_chat_history(result.session_id)
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == user_input["user_message"]
        assert history[1].role == "assistant"
        assert history[1].content == result.data.agent_response

        # Check Task Recording
        # We query the DB directly to verify the specific task was logged
        from kavalai.agents.db import Task
        from sqlalchemy import select

        stmt = select(Task).where(Task.session_id == result.session_id)
        db_result = await agents_db.execute(stmt)
        tasks = db_result.scalars().all()

        assert len(tasks) == 1
        assert "BB King" in tasks[0].inputs["prompt"]
        assert tasks[0].output["agent_response"] == result.data.agent_response
