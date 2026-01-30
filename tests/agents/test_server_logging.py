import unittest
from unittest.mock import patch, MagicMock
from kavalai.agents.server import run_agent_server


class TestServerLogging(unittest.TestCase):
    @patch("kavalai.agents.server.env")
    @patch("kavalai.agents.server.Workflow")
    @patch("kavalai.agents.server.db_manager")
    @patch("kavalai.agents.server.uvicorn.run")
    @patch("kavalai.agents.server.logger")
    def test_run_agent_server_logging(
        self, mock_logger, mock_uvicorn, mock_db_manager, mock_workflow, mock_env
    ):
        # Setup mocks
        mock_env.side_effect = lambda key, default=None: {
            "KAVALAI_AGENT_WORKFLOW_PATH": "test_path.yaml",
            "KAVALAI_DB_URI": "postgresql://user:password@localhost/dbname",
            "KAVALAI_DB_SCHEMA": "test_schema",
            "KAVALAI_AGENT_BASIC_AUTH_USER": "admin",
            "KAVALAI_AGENT_BASIC_AUTH_USER_PASSWORD": "secret_password",
        }.get(key, default)

        mock_env.str.side_effect = lambda key, default=None: {
            "KAVALAI_AGENT_WORKFLOW_PATH": "test_path.yaml",
            "KAVALAI_AGENT_HOST": "0.0.0.0",
            "KAVALAI_AGENT_BASIC_AUTH_USER": "admin",
            "KAVALAI_AGENT_BASIC_AUTH_USER_PASSWORD": "secret_password",
        }.get(key, default)

        mock_env.int.side_effect = lambda key, default=None: {
            "KAVALAI_AGENT_PORT": 10000,
        }.get(key, default)

        mock_env.bool.return_value = False

        mock_workflow_instance = MagicMock()
        mock_workflow_instance.workflow_model.name = "test_agent"
        mock_workflow_instance.get_data_type.return_value = str
        mock_workflow.from_yaml_path.return_value = mock_workflow_instance

        # Run the server
        run_agent_server()

        # Check logs
        info_logs = [call.args[0] for call in mock_logger.info.call_args_list]
        warning_logs = [call.args[0] for call in mock_logger.warning.call_args_list]

        print(f"Captured INFO logs: {info_logs}")
        print(f"Captured WARNING logs: {warning_logs}")

        # Assertions
        self.assertTrue(any("Database URI:" in log for log in info_logs))
        self.assertTrue(any("Database Schema: test_schema" in log for log in info_logs))
        self.assertTrue(
            any("Basic Auth configured for user: admin" in log for log in info_logs)
        )
        self.assertTrue(any("Basic Auth password:" in log for log in info_logs))

        # Verify masking
        db_log = [log for log in info_logs if "Database URI:" in log][0]
        self.assertIn("postgresql://user:***@localhost/dbname", db_log)
        self.assertNotIn("password", db_log)

        auth_log = [log for log in info_logs if "Basic Auth password:" in log][0]
        self.assertIn("Basic Auth password: ***", auth_log)
        self.assertNotIn("secret_password", auth_log)

    @patch("kavalai.agents.server.env")
    @patch("kavalai.agents.server.Workflow")
    @patch("kavalai.agents.server.db_manager")
    @patch("kavalai.agents.server.uvicorn.run")
    @patch("kavalai.agents.server.logger")
    def test_run_agent_server_no_auth_warning(
        self, mock_logger, mock_uvicorn, mock_db_manager, mock_workflow, mock_env
    ):
        # Setup mocks with NO auth
        mock_env.side_effect = lambda key, default=None: {
            "KAVALAI_AGENT_WORKFLOW_PATH": "test_path.yaml",
            "KAVALAI_DB_URI": "postgresql://user:password@localhost/dbname",
            "KAVALAI_DB_SCHEMA": "test_schema",
            "KAVALAI_AGENT_BASIC_AUTH_USER": "",
            "KAVALAI_AGENT_BASIC_AUTH_USER_PASSWORD": "",
        }.get(key, default)

        mock_env.str.side_effect = lambda key, default=None: {
            "KAVALAI_AGENT_WORKFLOW_PATH": "test_path.yaml",
            "KAVALAI_AGENT_HOST": "0.0.0.0",
            "KAVALAI_AGENT_BASIC_AUTH_USER": "",
            "KAVALAI_AGENT_BASIC_AUTH_USER_PASSWORD": "",
        }.get(key, default)

        mock_env.int.side_effect = lambda key, default=None: {
            "KAVALAI_AGENT_PORT": 10000,
        }.get(key, default)

        mock_env.bool.return_value = False

        mock_workflow_instance = MagicMock()
        mock_workflow_instance.workflow_model.name = "test_agent"
        mock_workflow_instance.get_data_type.return_value = str
        mock_workflow.from_yaml_path.return_value = mock_workflow_instance

        # Run the server
        run_agent_server()

        # Check warning log
        warning_logs = [call.args[0] for call in mock_logger.warning.call_args_list]
        self.assertTrue(
            any("Basic Auth is NOT configured" in log for log in warning_logs)
        )


if __name__ == "__main__":
    unittest.main()
