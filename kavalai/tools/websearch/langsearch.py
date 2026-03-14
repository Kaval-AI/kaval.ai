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
import logging
from typing import Optional, List, Literal

import httpx
from pydantic import BaseModel, Field

from kavalai.functionkernel import pythontool

logger = logging.getLogger(__name__)

# Constants
LANGSEARCH_API_DOMAIN = "https://api.langsearch.com"
LANGSEARCH_API_ENDPOINT = f"{LANGSEARCH_API_DOMAIN}/v1/web-search"


# Models
class QueryContext(BaseModel):
    originalQuery: str


class WebPageValue(BaseModel):
    id: str
    name: str
    url: str
    displayUrl: str
    snippet: str
    summary: Optional[str] = None
    datePublished: Optional[str] = None
    dateLastCrawled: Optional[str] = None


class WebPages(BaseModel):
    webSearchUrl: Optional[str] = None
    totalEstimatedMatches: Optional[int] = None
    value: List[WebPageValue]
    someResultsRemoved: Optional[bool] = None


class SearchResponse(BaseModel):
    model_config = {"populate_by_name": True}
    search_type: str = Field(alias="_type", default="SearchResponse")
    queryContext: QueryContext
    webPages: WebPages


class LangSearchRequest(BaseModel):
    query: str
    freshness: Optional[
        Literal["oneDay", "oneWeek", "oneMonth", "oneYear", "noLimit"]
    ] = "noLimit"
    summary: Optional[bool] = False
    count: Optional[int] = Field(default=10, ge=1, le=10)


@pythontool
def langsearch_web_search(
    query: str,
    freshness: Optional[
        Literal["oneDay", "oneWeek", "oneMonth", "oneYear", "noLimit"]
    ] = "noLimit",
    summary: bool = False,
    count: int = 10,
    api_key: Optional[str] = None,
) -> SearchResponse:
    """
    Perform a web search using LangSearch API.

    Args:
        query: The user's search query.
        freshness: Specifies the time range for search results.
            - oneDay: Results from the past 24 hours.
            - oneWeek: Results from the past week.
            - oneMonth: Results from the past month.
            - oneYear: Results from the past year.
            - noLimit: No time filter (default).
        summary: Whether to show long text summaries for results.
        count: The number of results to return (1-10, default 10).
        api_key: Optional API key. If not provided, will use LANGSEARCH_API_KEY environment variable.
    """
    key = api_key or os.environ.get("LANGSEARCH_API_KEY")
    if not key:
        raise ValueError(
            "LangSearch API key not provided. Set LANGSEARCH_API_KEY environment variable."
        )

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    request_data = LangSearchRequest(
        query=query, freshness=freshness, summary=summary, count=count
    ).model_dump(exclude_none=True)

    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            LANGSEARCH_API_ENDPOINT, json=request_data, headers=headers
        )
        response.raise_for_status()
        data = response.json()
        if "data" in data and isinstance(data["data"], dict):
            return SearchResponse.model_validate(data["data"])
        return SearchResponse.model_validate(data)
