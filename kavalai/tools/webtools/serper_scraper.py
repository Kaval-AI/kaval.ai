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
from typing import Optional, Dict, Any

import httpx
from pydantic import BaseModel

from kavalai.functionkernel import pythontool

logger = logging.getLogger(__name__)

# Constants
SERPER_SCRAPE_ENDPOINT = "https://scrape.serper.dev"


# Models
class SerperScrapeResponse(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None
    credits: Optional[int] = None


@pythontool
def serper_scrape_url(
    url: str,
) -> SerperScrapeResponse:
    """
    Scrape text content from a URL using Serper.dev Scrape API.

    Args:
        url: The website URL to scrape.
    """
    key = os.environ.get("SERPER_API_KEY")
    if not key:
        raise ValueError(
            "Serper API key not provided. Set SERPER_API_KEY environment variable."
        )

    headers = {
        "X-API-KEY": key,
        "Content-Type": "application/json",
    }
    payload = {"url": url}

    with httpx.Client(timeout=30.0) as client:
        response = client.post(SERPER_SCRAPE_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return SerperScrapeResponse.model_validate(data)
