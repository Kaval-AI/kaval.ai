import html
from typing import Optional

import feedparser
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

# Initialize FastMCP server
mcp = FastMCP("RSS-Parser", port=10000)


class RssFeedItem(BaseModel):
    title: Optional[str] = None
    link: Optional[str] = None
    summary: Optional[str] = None


class Feed(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    items: list[RssFeedItem] = []


@mcp.tool()
def get_rss_feed(url: str, max_results: int = 5) -> Feed:
    """Fetches and parses an RSS or Atom feed from a given URL.

    Args:
        url: The valid URL of the RSS feed.
        max_results: Number of recent articles to return (default 5).
    """
    try:
        feed = feedparser.parse(url)
        if not feed.entries:
            return Feed()
        result = Feed(title=feed.feed.get("title"), url=url)
        for entry in feed.entries[:max_results]:
            title = entry.get("title", "No Title")
            link = entry.get("link", "No Link")
            summary = html.unescape(entry.get("summary", "No summary available."))

            result.items.append(RssFeedItem(title=title, link=link, summary=summary))
        return result
    except Exception as e:
        raise Exception(f"Error parsing feed: {str(e)}")


if __name__ == "__main__":
    mcp.run(transport="sse")
