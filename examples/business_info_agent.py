"""Example: a v2 Agent that researches a business using Serper web search + scraping."""

import asyncio
import os
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.json import JSON

from kavalai import FunctionKernel, make_client
from kavalai.agents.run_context import RunContext
from kavalai.agents.v2.agent import Agent
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


async def main():
    if not os.environ.get("SERPER_API_KEY"):
        print("Please set SERPER_API_KEY environment variable.")
        return

    # 1. Register the tools the agent may call (addressed as python://<name>).
    kernel = FunctionKernel()
    kernel.register_python_tool("websearch.serper_web_search", serper_web_search)
    kernel.register_python_tool("webtools.serper_scrape_url", serper_scrape_url)

    # 2. Seed the run context with the query the agent will research.
    run_context = RunContext()
    run_context.data["input"] = {"business_query": "Kaval AI"}

    # 3. Build the v2 agent on a capable model and run the multi-step loop.
    agent = Agent(
        llm_client=make_client("gemini/gemini-3.1-pro-preview"),
        kernel=kernel,
        run_context=run_context,
    )

    task = (
        "Find information about the business named in input.business_query using the "
        "web search and scrape tools, then fill in the business information form."
    )
    console.print(f"Running agent for task: [italic]{task}[/italic]")
    console.print("-" * 50)

    result = await agent.prompt(task, response_model=BusinessInfo, max_steps=10)

    if result:
        console.print(
            "\n[bold green]Successfully found business information:[/bold green]"
        )
        console.print(JSON(result.model_dump_json()))
    else:
        console.print(
            "\n[bold red]Agent failed to produce the requested output.[/bold red]"
        )

    await kernel.close()


if __name__ == "__main__":
    asyncio.run(main())
