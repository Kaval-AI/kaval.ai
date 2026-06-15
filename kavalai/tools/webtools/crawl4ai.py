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

from typing import Optional, Dict, Any

from loguru import logger
from pydantic import BaseModel

from kavalai.functionkernel import pythontool


class Crawl4aiResponse(BaseModel):
    url: str
    success: bool
    markdown: Optional[str] = None
    html: Optional[str] = None
    status_code: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@pythontool
async def crawl_url(
    url: str,
    include_html: bool = False,
    bypass_cache: bool = False,
    timeout: float = 60.0,
) -> Crawl4aiResponse:
    """
    Crawl a web page and return its content as clean Markdown using Crawl4AI.

    Args:
        url: The website URL to crawl.
        include_html: Whether to also return the cleaned HTML (default False).
        bypass_cache: Whether to bypass the crawler cache and fetch fresh content (default False).
        timeout: Page load timeout in seconds (default 60.0).
    """
    # Imported lazily so the optional 'tools' dependency is only required at call time.
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

    browser_config = BrowserConfig(headless=True)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS if bypass_cache else CacheMode.ENABLED,
        page_timeout=int(timeout * 1000),
    )

    logger.info(f"Crawling URL with Crawl4AI: {url}")
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)

    markdown = None
    if result.markdown is not None:
        markdown = getattr(result.markdown, "raw_markdown", str(result.markdown))

    return Crawl4aiResponse(
        url=result.url,
        success=result.success,
        markdown=markdown,
        html=result.cleaned_html if include_html else None,
        status_code=result.status_code,
        metadata=result.metadata,
        error_message=result.error_message,
    )
