import sys
import json
import asyncio


async def main():
    while True:
        line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        try:
            request = json.loads(line)
            if "method" in request:
                if request["method"] == "initialize":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "test-error", "version": "1.0"},
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
                                    "name": "fail_tool",
                                    "description": "Fails",
                                    "inputSchema": {"type": "object"},
                                }
                            ]
                        },
                    }
                elif request["method"] == "tools/call":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": {
                            "content": [
                                {"type": "text", "text": "Something went wrong"}
                            ],
                            "isError": True,
                        },
                    }
                else:
                    response = {"jsonrpc": "2.0", "id": request.get("id"), "result": {}}
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
