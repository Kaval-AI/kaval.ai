import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client


async def main():
    # The URL to your Streamable HTTP / SSE server
    url = "http://localhost:10000/sse"

    # sse_client handles the Streamable HTTP protocol automatically
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            # Step 1: Initialize the connection
            await session.initialize()

            # Step 2: List the tools to verify connection
            tools = await session.list_tools()
            print(f"Server found with {len(tools.tools)} tools.")

            # Step 3: Call your RSS tool
            print("Fetching RSS feed...")
            result = await session.call_tool(
                "get_rss_feed", arguments={"url": "https://news.ycombinator.com/rss"}
            )

            # Result is returned as a list of content objects
            print(result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
