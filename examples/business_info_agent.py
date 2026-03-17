"""
Example script demonstrating the use of PlanningAgent with Serper web search and scrape tools.
"""

import asyncio
import os
import json
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
from rich.console import Console
from rich.json import JSON

from kavalai.agents.planning_agent import PlanningAgent
from kavalai.agents.run_context import RunContext
from kavalai.functionkernel import FunctionKernel
from kavalai.llm_clients.llm_client import LLMClient
from kavalai.llm_clients.common import Streamer, StreamContent
from kavalai.tools.websearch.serper import serper_web_search
from kavalai.tools.webtools.serper_scraper import serper_scrape_url


class BusinessInfo(BaseModel):
    """Information about a business found online."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="The legal or trading name of the business.")
    address: Optional[str] = Field(description="The physical address of the business.")
    website: Optional[str] = Field(description="The official website URL.")
    phone: Optional[str] = Field(description="Contact phone number.")
    owners: Optional[str] = Field(description="The owners of the business.")
    description: str = Field(
        description="A brief description of what the business does."
    )
    industry: Optional[str] = Field(
        description="The industry the business operates in."
    )


console = Console()


async def stream_consumer(queue: asyncio.Queue):
    """Consume streamed data and print it to the console using rich."""
    while True:
        try:
            message_json = await queue.get()
            if message_json is None:
                queue.task_done()
                break

            message = StreamContent.model_validate_json(message_json)

            if message.type == "partial":
                # Turn off delta streaming by not printing partial dots
                pass
            elif message.type == "complete":
                console.print(
                    f"\n[bold green][Stream Complete][/bold green] [cyan]{message.name}[/cyan]:"
                )
                try:
                    # If it's valid JSON, format it
                    json_data = json.loads(message.value)
                    console.print(JSON(json.dumps(json_data)))
                except json.JSONDecodeError:
                    # If it's not JSON, print as is
                    console.print(message.value)

            queue.task_done()
        except Exception as e:
            console.print(f"\n[bold red]Error in stream consumer:[/bold red] {e}")
            queue.task_done()


async def main():
    # Ensure API keys are set
    if not os.environ.get("SERPER_API_KEY"):
        print("Please set SERPER_API_KEY environment variable.")
        return
    # 1. Initialize the FunctionKernel and register tools
    kernel = FunctionKernel()
    kernel.register_python_tool(
        "python://websearch.serper_web_search", serper_web_search
    )
    kernel.register_python_tool(
        "python://webtools.serper_scrape_url", serper_scrape_url
    )

    # 2. Initialize the LLM Client
    # Using a capable model for planning and extraction
    llm_client = LLMClient(model="openai/gpt-5.4")

    # 3. Initialize RunContext, Streamer and PlanningAgent
    run_context = RunContext()
    input_data = {"business_query": "Kaval AI"}

    stream_queue = asyncio.Queue()
    streamer = Streamer(name="BusinessInfoAgent", queue=stream_queue)

    # Start stream consumer
    consumer_task = asyncio.create_task(stream_consumer(stream_queue))

    agent = PlanningAgent(
        kernel=kernel,
        run_context=run_context,
        llm_client=llm_client,
        input_data=input_data,
        response_model=BusinessInfo,
        temperature=0,
        streamer=streamer,
    )

    # 4. Run the agent with a task
    task = (
        "Find information about a business online using web search and scrape tools. "
        "Search for the business mentioned in the input data, then scrape its website "
        "or other reliable sources to fill in the business information form."
    )

    console.print(f"Starting PlanningAgent for task: [italic]{task}[/italic]")
    console.print(f"Input: [bold blue]{input_data['business_query']}[/bold blue]")
    console.print("-" * 50)

    result = await agent.run(
        task=task,
        max_iterations=10,
        chat_history=[{"role": "user", "content": "Get me Kaval.AI profile."}],
    )

    if result:
        console.print(
            "\n[bold green]Successfully found business information:[/bold green]"
        )
        console.print(JSON(result.model_dump_json()))
    else:
        console.print(
            "\n[bold red]Agent failed to produce the requested output.[/bold red]"
        )

    # Stop stream consumer
    await stream_queue.put(None)
    await consumer_task
    await kernel.close()


if __name__ == "__main__":
    asyncio.run(main())
