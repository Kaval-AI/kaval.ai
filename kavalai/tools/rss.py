import html
import os
import secrets
from typing import Annotated, Optional

import feedparser
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

app = FastAPI(title="RSS-Parser", version="1.0.0")
security = HTTPBasic()


def validate_auth(credentials: Annotated[HTTPBasicCredentials, Depends(security)]):
    expected_username = os.environ.get("RSS_AUTH_USER", "admin")
    expected_password = os.environ.get("RSS_AUTH_PASSWORD", "password")

    is_correct_username = secrets.compare_digest(
        credentials.username, expected_username
    )
    is_correct_password = secrets.compare_digest(
        credentials.password, expected_password
    )

    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


class RssFeedItem(BaseModel):
    title: Optional[str] = None
    link: Optional[str] = None
    summary: Optional[str] = None


class Feed(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    items: list[RssFeedItem] = []


@app.get("/get_rss_feed", response_model=Feed)
def get_rss_feed(
    url: str,
    max_results: int = 5,
    username: str = Depends(validate_auth),
) -> Feed:
    """Fetches and parses an RSS or Atom feed from a given URL.

    Args:
        url: The valid URL of the RSS feed.
        max_results: Number of recent articles to return (default 5).
        username: Authenticated username.
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error parsing feed: {str(e)}",
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
