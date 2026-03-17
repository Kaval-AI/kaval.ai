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
from typing import Optional, List

import httpx
from pydantic import BaseModel

from kavalai.functionkernel import pythontool


# Constants
SERPER_API_ENDPOINT = "https://google.serper.dev/search"


# Models
class SearchParameters(BaseModel):
    q: str
    gl: Optional[str] = None
    hl: Optional[str] = None
    tbs: Optional[str] = None
    page: Optional[int] = None
    type: Optional[str] = None
    engine: Optional[str] = None


class OrganicResult(BaseModel):
    title: str
    link: str
    snippet: str
    position: int
    date: Optional[str] = None


class SerperSearchResponse(BaseModel):
    searchParameters: SearchParameters
    organic: List[OrganicResult]
    credits: Optional[int] = None


class SerperSearchRequest(BaseModel):
    q: str
    gl: Optional[str] = None
    hl: Optional[str] = None
    tbs: Optional[str] = None
    page: Optional[int] = 1


@pythontool
def serper_web_search(
    query: str,
    country: Optional[str] = None,
    language: Optional[str] = None,
    date_range: Optional[str] = None,
    page: int = 1,
) -> SerperSearchResponse:
    """
    Perform a web search using Serper.dev API (Google Search).

    Args:
        query: The user's search query.
        country: Two-letter country code (e.g., 'us', 'ee').
        language: Two-letter language code (e.g., 'en', 'et').
        date_range: Date range for search results (e.g., 'qdr:h' for hour, 'qdr:d' for day, 'qdr:w' for week, 'qdr:m' for month, 'qdr:y' for year).
        page: Page number of results (default 1).
    """
    key = os.environ.get("SERPER_API_KEY")
    if not key:
        raise ValueError(
            "Serper API key not provided. Set SERPER_API_KEY environment variable."
        )

    headers = {"X-API-KEY": key, "Content-Type": "application/json"}

    request_data = SerperSearchRequest(
        q=query, gl=country, hl=language, tbs=date_range, page=page
    ).model_dump(exclude_none=True)

    with httpx.Client(timeout=30.0) as client:
        response = client.post(SERPER_API_ENDPOINT, json=request_data, headers=headers)
        response.raise_for_status()
        data = response.json()
        return SerperSearchResponse.model_validate(data)
