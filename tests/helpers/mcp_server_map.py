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
                            "capabilities": {},
                            "serverInfo": {"name": "test-map", "version": "1.0"},
                        },
                    }
                elif request["method"] == "tools/list":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": {
                            "tools": [
                                {
                                    "name": "primitive",
                                    "description": "Returns primitive",
                                    "inputSchema": {"type": "object", "properties": {}},
                                }
                            ]
                        },
                    }
                elif request["method"] == "tools/call":
                    # Return primitive 42
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": {
                            "content": [{"type": "text", "text": "42"}],
                            "isError": False,
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
