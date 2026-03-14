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
from unittest.mock import patch, MagicMock
import httpx
from kavalai.tools.webtools.serper_scraper import (
    serper_scrape_url,
    SerperScrapeResponse,
)


def test_serper_scrape_url_success():
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "text": "Scraped content",
        "metadata": {"title": "Example"},
        "credits": 1,
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.Client.post") as mock_post:
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"SERPER_API_KEY": "test_key"}):
            response = serper_scrape_url(url="https://example.com")

            assert isinstance(response, SerperScrapeResponse)
            assert response.text == "Scraped content"
            assert response.metadata == {"title": "Example"}
            assert response.credits == 1

            mock_post.assert_called_once_with(
                "https://scrape.serper.dev",
                json={"url": "https://example.com"},
                headers={"X-API-KEY": "test_key", "Content-Type": "application/json"},
            )


def test_serper_scrape_url_no_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Serper API key not provided"):
            serper_scrape_url(url="https://example.com")


def test_serper_scrape_url_http_error():
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Error", request=MagicMock(), response=mock_response
    )

    with patch("httpx.Client.post") as mock_post:
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"SERPER_API_KEY": "test_key"}):
            with pytest.raises(httpx.HTTPStatusError):
                serper_scrape_url(url="https://example.com")


@pytest.mark.skipif(
    not os.environ.get("SERPER_API_KEY"), reason="SERPER_API_KEY not defined"
)
def test_serper_scrape_url_integration():
    """
    Real integration test for serper_scrape_url tool.
    This test will only run if SERPER_API_KEY is defined in the environment.
    """
    url = "https://example.com"
    response = serper_scrape_url(url=url)

    assert isinstance(response, SerperScrapeResponse)
    assert response.text is not None
    assert "Example Domain" in response.text
    assert response.credits is not None
    assert response.credits > 0
