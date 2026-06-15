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

import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from kavalai.tools.webtools.crawl4ai import (
    crawl_url,
    Crawl4aiResponse,
)


def _make_crawler_mock(result):
    """Build a mock AsyncWebCrawler usable as an async context manager."""
    crawler = MagicMock()
    crawler.arun = AsyncMock(return_value=result)
    crawler.__aenter__ = AsyncMock(return_value=crawler)
    crawler.__aexit__ = AsyncMock(return_value=False)
    crawler_cls = MagicMock(return_value=crawler)
    return crawler_cls, crawler


@pytest.mark.asyncio
async def test_crawl_url_success():
    markdown = MagicMock()
    markdown.raw_markdown = "# Example"

    result = MagicMock()
    result.url = "https://example.com"
    result.success = True
    result.markdown = markdown
    result.cleaned_html = "<h1>Example</h1>"
    result.status_code = 200
    result.metadata = {"title": "Example"}
    result.error_message = None

    crawler_cls, crawler = _make_crawler_mock(result)

    with patch("crawl4ai.AsyncWebCrawler", crawler_cls):
        response = await crawl_url(url="https://example.com")

    assert isinstance(response, Crawl4aiResponse)
    assert response.url == "https://example.com"
    assert response.success is True
    assert response.markdown == "# Example"
    # HTML omitted by default.
    assert response.html is None
    assert response.status_code == 200
    assert response.metadata == {"title": "Example"}
    crawler.arun.assert_awaited_once()


@pytest.mark.asyncio
async def test_crawl_url_include_html_and_bypass_cache():
    result = MagicMock()
    result.url = "https://example.com"
    result.success = True
    result.markdown = None
    result.cleaned_html = "<h1>Example</h1>"
    result.status_code = 200
    result.metadata = None
    result.error_message = None

    crawler_cls, crawler = _make_crawler_mock(result)

    with patch("crawl4ai.AsyncWebCrawler", crawler_cls):
        response = await crawl_url(
            url="https://example.com", include_html=True, bypass_cache=True
        )

    assert response.markdown is None
    assert response.html == "<h1>Example</h1>"

    # Bypass cache must be reflected in the run config.
    from crawl4ai import CacheMode

    _, kwargs = crawler.arun.call_args
    assert kwargs["config"].cache_mode == CacheMode.BYPASS


@pytest.mark.asyncio
async def test_crawl_url_failure():
    result = MagicMock()
    result.url = "https://example.com"
    result.success = False
    result.markdown = None
    result.cleaned_html = None
    result.status_code = 404
    result.metadata = None
    result.error_message = "Not found"

    crawler_cls, _ = _make_crawler_mock(result)

    with patch("crawl4ai.AsyncWebCrawler", crawler_cls):
        response = await crawl_url(url="https://example.com")

    assert response.success is False
    assert response.markdown is None
    assert response.error_message == "Not found"


@pytest.mark.asyncio
async def test_crawl_url_markdown_without_raw_attr():
    # Some crawl4ai versions return a plain string for markdown.
    result = MagicMock()
    result.url = "https://example.com"
    result.success = True
    result.markdown = "plain markdown"
    result.cleaned_html = None
    result.status_code = 200
    result.metadata = None
    result.error_message = None

    crawler_cls, _ = _make_crawler_mock(result)

    with patch("crawl4ai.AsyncWebCrawler", crawler_cls):
        response = await crawl_url(url="https://example.com")

    assert response.markdown == "plain markdown"


@pytest.mark.skipif(
    not os.environ.get("CRAWL4AI_INTEGRATION"),
    reason="CRAWL4AI_INTEGRATION not defined",
)
@pytest.mark.asyncio
async def test_crawl_url_integration():
    """
    Real integration test for crawl_url tool.
    Only runs if CRAWL4AI_INTEGRATION is defined and a browser is installed.
    """
    response = await crawl_url(url="https://example.com")

    assert isinstance(response, Crawl4aiResponse)
    assert response.success is True
    assert response.markdown is not None
    assert "Example Domain" in response.markdown
