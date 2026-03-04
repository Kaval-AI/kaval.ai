from unittest.mock import patch, MagicMock
from kavalai.tools.cli_chat import main


@patch("sys.argv", ["cli_chat.py", "--url", "http://localhost:8000"])
@patch("kavalai.tools.cli_chat.AgentClient")
@patch("kavalai.tools.cli_chat.Prompt.ask", return_value="exit")
@patch("kavalai.tools.cli_chat.Console.print")
async def test_cli_chat_url_with_port(mock_print, mock_prompt, MockClient):
    mock_client_instance = MockClient.return_value

    async def mock_discover():
        return None

    mock_client_instance.discover_schemas = MagicMock(side_effect=mock_discover)

    # Mock input fields for schemas
    mock_client_instance.input_schema.model_fields = {"user_message": MagicMock()}
    mock_client_instance.output_schema.model_fields = {"agent_response": MagicMock()}

    # Run main and catch SystemExit if any
    try:
        await main()
    except SystemExit:
        pass

    # Verify AgentClient was initialized with the correct URL
    MockClient.assert_called_once()
    args, _ = MockClient.call_args
    assert args[0] == "http://localhost:8000"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
