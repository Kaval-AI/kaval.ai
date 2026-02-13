"""
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import json
import sys
from argparse import ArgumentParser

from rich.console import Console
from rich.prompt import Prompt

from kavalai.agents.client import AgentClient
from kavalai.llm_clients.common import StreamContent

console = Console()


async def main():
    parser = ArgumentParser(description="CLI Chat tool for Kaval.AI agents.")
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Agent server URL (e.g. http://localhost)",
    )
    parser.add_argument("--port", type=int, required=True, help="Agent server port")
    parser.add_argument("--user", type=str, help="Basic auth username")
    parser.add_argument("--password", type=str, help="Basic auth password")

    args = parser.parse_args()

    base_url = f"{args.url.rstrip('/')}:{args.port}"
    client = AgentClient(base_url, args.user, args.password)

    console.print(f"[bold green]Connecting to agent at {base_url}...[/bold green]")
    try:
        await client.discover_schemas()
    except Exception as e:
        console.print(
            f"[bold red]Failed to connect or discover schemas: {e}[/bold red]"
        )
        sys.exit(1)

    console.print(
        "[bold blue]Connected! Type 'exit' or 'quit' to end the chat.[/bold blue]"
    )
    console.print(f"Input schema: {client.input_schema.model_fields.keys()}")
    console.print(f"Output schema: {client.output_schema.model_fields.keys()}")

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
            if user_input.lower() in ("exit", "quit"):
                break

            if not user_input.strip():
                continue

            # Assuming the agent expects a 'user_message' field based on the issue description.
            # If the schema is different, this might need more complex input handling.
            if "user_message" not in client.input_schema.model_fields:
                console.print(
                    f"[bold red]Error: Agent input schema does not have 'user_message' field. Fields: {client.input_schema.model_fields.keys()}[/bold red]"
                )
                break

            data = client.input_schema(user_message=user_input)

            console.print("[bold yellow]Agent:[/bold yellow] ", end="")
            last_response = ""
            async for line in client.stream_agent(data):
                try:
                    stream_content = StreamContent.model_validate_json(line)
                    if stream_content.type == "partial":
                        try:
                            # Try to parse as JSON to extract agent_response
                            val_dict = json.loads(stream_content.value)
                            if "agent_response" in val_dict:
                                new_response = val_dict["agent_response"]
                                if new_response.startswith(last_response):
                                    diff = new_response[len(last_response) :]
                                    console.print(diff, end="")
                                    last_response = new_response
                                else:
                                    # If it doesn't start with last_response, it might be a new part or complete replacement
                                    console.print(new_response, end="")
                                    last_response = new_response
                            elif not val_dict:
                                # Skip empty JSON objects like {}
                                pass
                            else:
                                console.print(stream_content.value, end="")
                        except json.JSONDecodeError:
                            # Not JSON, just print it
                            console.print(stream_content.value, end="")
                    elif stream_content.type == "complete":
                        break
                except Exception:
                    # Ignore parsing errors for now
                    pass
            console.print()

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")

    console.print("[bold blue]Goodbye![/bold blue]")


if __name__ == "__main__":
    asyncio.run(main())
