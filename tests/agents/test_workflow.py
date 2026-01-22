import pytest
import httpx
from unittest.mock import patch
from kavalai.agents.workflow import Workflow, WorkflowModel
from kavalai.agents.agent_service import AgentService
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import uvicorn
import threading
import time
import os
import yaml

# Simple Mock RSS Server using FastAPI
rss_app = FastAPI()
security = HTTPBasic()


def validate_auth(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != "admin" or credentials.password != "password":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


@rss_app.get("/get_rss_feed")
def get_rss_feed(url: str, username: str = Depends(validate_auth)):
    return {
        "title": "Mock Feed",
        "url": url,
        "items": [
            {
                "title": "Mock News 1",
                "link": "http://example.com/1",
                "summary": "Summary 1",
            },
            {
                "title": "Mock News 2",
                "link": "http://example.com/2",
                "summary": "Summary 2",
            },
        ],
    }


def run_mock_server():
    uvicorn.run(rss_app, host="127.0.0.1", port=13005, log_level="error")


@pytest.fixture(scope="module")
def rss_server():
    thread = threading.Thread(target=run_mock_server, daemon=True)
    thread.start()
    # Wait for server to start
    for _ in range(50):
        try:
            httpx.get(
                "http://127.0.0.1:13005/get_rss_feed",
                params={"url": "test"},
                auth=("admin", "password"),
            )
            break
        except httpx.ConnectError:
            time.sleep(0.1)
    yield


@pytest.mark.asyncio
class TestWorkflowWithRestTools:
    async def test_herold_workflow_with_rest_tool(
        self, agents_db, tmp_path, monkeypatch, rss_server
    ):
        """
        Tests the Herold workflow using a real (mocked) REST server for tools.
        """
        # 1. Setup
        service = AgentService(agents_db)

        # Load herold.yaml
        herold_yaml_path = os.path.join(
            os.path.dirname(__file__), "../../demo_agents/herold.yaml"
        )
        with open(herold_yaml_path, "r") as f:
            herold_yaml_content = f.read()

        # Override the rss server URL and rest servers in the YAML for the test if needed
        # (herold.yaml already uses http://localhost:13001, but the mock server is at 13005)
        yaml_dict = yaml.safe_load(herold_yaml_content)
        for rs in yaml_dict.get("rest_servers", []):
            if rs["name"] == "rss":
                rs["url"] = "http://127.0.0.1:13005"

        wf = Workflow(WorkflowModel(**yaml_dict))
        wf.agent_service = service

        # Ensure LLM profile exists
        profile_name = "openai"
        profile_dir = tmp_path / "llm_profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        # Note: profile name in YAML has a slash, we need to handle that or change it.
        # Workflow.from_yaml loads the profile name from the YAML.
        # kavalai.agents.agent_service.load_profile_from_path joins LLM_PROFILES_PATH and name + ".yaml"

        # Create directories for the profile name if it contains slashes
        profile_file_path = profile_dir / (profile_name + ".yaml")
        profile_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(profile_file_path, "w") as f:
            yaml.dump(
                {
                    "name": profile_name,
                    "provider": "openai",
                    "model_name": "gpt-5.2",
                },
                f,
            )

        monkeypatch.setenv("LLM_PROFILES_PATH", str(profile_dir))

        user_input = {"user_message": "What's the news today?"}

        # Mock chat_completion_with_stats
        output_type = wf.get_data_type("output")
        mock_response = output_type(agent_response="The news today is great.")

        with patch(
            "kavalai.agents.workflow.chat_completion_with_stats"
        ) as mock_chat_completion:
            mock_chat_completion.return_value = mock_response

            # 2. Execution
            result = await wf.run(input_data=user_input)

        # 3. Validation
        assert result.session_id is not None
        assert result.data is not None
        assert result.data.agent_response == "The news today is great."

        # Verify that the tools were called and data is in the run context
        # In herold.yaml, the data type is named rss_feed and used as ycombinator_news
        news_data = result.run_context.data["ycombinator_news"]
        assert news_data.title == "Mock Feed"
        assert len(news_data.items) == 2

        # 4. Validation: Database Records
        agent = await service.get_or_create_agent(name="Herold news agent")
        assert "news" in agent.description.lower()

        # Check Task Recording (should have 2 tool tasks and 1 prompt task)
        from kavalai.agents.db import Task as DBTask
        from sqlalchemy import select

        stmt = (
            select(DBTask)
            .where(DBTask.session_id == result.session_id)
            .order_by(DBTask.created_at)
        )
        db_result = await agents_db.execute(stmt)
        tasks = db_result.scalars().all()

        assert len(tasks) == 3
        # First task: get_rss_feed (ycombinator_news)
        assert tasks[0].inputs["tool"] == "get_rss_feed"
        # Last task: Compute the response
        assert "Agent task: You are Herold" in tasks[2].inputs["prompt"]


def test_workflow_model_embedding_name():
    from kavalai.agents.workflow import WorkflowModel

    yaml_data = """
name: Test Agent
description: A test agent
llm_profile_name: openai
llm_embedding_name: openai-embed
data_types:
  input: {type: object, properties: {}}
tasks: []
"""
    model = WorkflowModel(**yaml.safe_load(yaml_data))
    assert model.llm_embedding_name == "openai-embed"

    yaml_no_embed = """
name: Test Agent
description: A test agent
llm_profile_name: openai
data_types:
  input: {type: object, properties: {}}
tasks: []
"""
    model_no_embed = WorkflowModel(**yaml.safe_load(yaml_no_embed))
    assert model_no_embed.llm_embedding_name is None
