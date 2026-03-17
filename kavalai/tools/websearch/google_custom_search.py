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
from typing import Optional, List, Dict, Any

import httpx
from pydantic import BaseModel, Field

from kavalai.functionkernel import pythontool


# Constants
GOOGLE_CUSTOM_SEARCH_ENDPOINT = "https://www.googleapis.com/customsearch/v1"


# Models
class SearchItem(BaseModel):
    title: str
    link: str
    snippet: str
    displayLink: Optional[str] = None
    pagemap: Optional[Dict[str, Any]] = None


class SearchMetadata(BaseModel):
    totalResults: Optional[str] = None
    searchTime: Optional[float] = None


class CustomSearchResponse(BaseModel):
    items: List[SearchItem] = Field(default_factory=list)
    searchInformation: Optional[SearchMetadata] = None


@pythontool
def google_custom_search(
    query: str,
    api_key: Optional[str] = None,
    cx: Optional[str] = None,
    num: int = 10,
    **kwargs: Any,
) -> CustomSearchResponse:
    """
    Perform a web search using Google Custom Search JSON API.

    Args:
        query: The user's search query (q).
        api_key: Google API key (key). If not provided, will use GOOGLE_CUSTOM_SEARCH_API_KEY environment variable.
        cx: Programmable Search Engine ID (cx). If not provided, will use GOOGLE_CUSTOM_SEARCH_CX environment variable.
        num: Number of search results to return (1-10, default 10).
        **kwargs: Additional optional query parameters for the API.
    """
    key = api_key or os.environ.get("GOOGLE_CUSTOM_SEARCH_API_KEY")
    engine_id = cx or os.environ.get("GOOGLE_CUSTOM_SEARCH_CX")

    if not key:
        raise ValueError(
            "Google API key not provided. Set GOOGLE_CUSTOM_SEARCH_API_KEY environment variable."
        )
    if not engine_id:
        raise ValueError(
            "Programmable Search Engine ID (cx) not provided. Set GOOGLE_CUSTOM_SEARCH_CX environment variable."
        )

    params = {
        "key": key,
        "cx": engine_id,
        "q": query,
        "num": min(max(num, 1), 10),  # API allows 1-10 results per request
    }
    params.update(kwargs)

    with httpx.Client(timeout=30.0) as client:
        response = client.get(GOOGLE_CUSTOM_SEARCH_ENDPOINT, params=params)
        response.raise_for_status()
        data = response.json()
        return CustomSearchResponse.model_validate(data)
