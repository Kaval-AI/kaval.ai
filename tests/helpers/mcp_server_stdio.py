import sys
import json
import asyncio


async def main():
    try:
        while True:
            line = await asyncio.get_event_loop().run_in_executor(
                None, sys.stdin.readline
            )
            if not line:
                break
            request = json.loads(line)
            if "method" in request:
                if request["method"] == "initialize":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "test-stdio", "version": "1.0"},
                        },
                    }
                elif request["method"] == "notifications/initialized":
                    continue
                elif request["method"] == "tools/list":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": {
                            "tools": [
                                {
                                    "name": "test_tool",
                                    "description": "A test tool",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {"arg": {"type": "string"}},
                                    },
                                }
                            ]
                        },
                    }
                elif request["method"] == "tools/call":
                    tool_name = request["params"]["name"]
                    if tool_name == "test_tool":
                        content = {"name": "mcp_test", "value": 200, "result": "ok"}
                        response = {
                            "jsonrpc": "2.0",
                            "id": request["id"],
                            "result": {
                                "content": [
                                    {"type": "text", "text": json.dumps(content)}
                                ],
                                "isError": False,
                            },
                        }
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "error": {"code": -32601, "message": "Method not found"},
                    }

                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
    finally:
        # Wait a bit to ensure all data is flushed
        await asyncio.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(main())
