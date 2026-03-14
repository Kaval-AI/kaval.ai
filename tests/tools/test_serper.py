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
from unittest.mock import patch, MagicMock
import pytest
import httpx
from kavalai.tools.websearch.serper import serper_web_search, SerperSearchResponse


def test_serper_web_search_success():
    mock_response = {
        "searchParameters": {
            "q": "kaval ai",
            "gl": "us",
            "hl": "en",
            "tbs": "qdr:d",
            "page": 1,
            "type": "search",
            "engine": "google",
        },
        "organic": [
            {
                "title": "Kaval.AI - AI agent consultancy and development services company.",
                "link": "https://kaval.ai/",
                "snippet": "Kaval.AI. Agent development toolkit.",
                "position": 1,
            }
        ],
        "credits": 1,
    }

    with patch("httpx.Client.post") as mock_post:
        mock_post.return_value = MagicMock(spec=httpx.Response)
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status = MagicMock()

        result = serper_web_search(
            query="kaval ai",
            api_key="test_key",
            country="us",
            language="en",
            date_range="qdr:d",
            page=1,
        )

        assert isinstance(result, SerperSearchResponse)
        assert result.searchParameters.q == "kaval ai"
        assert len(result.organic) == 1
        assert (
            result.organic[0].title
            == "Kaval.AI - AI agent consultancy and development services company."
        )

        # Verify request parameters
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["q"] == "kaval ai"
        assert kwargs["json"]["gl"] == "us"
        assert kwargs["json"]["hl"] == "en"
        assert kwargs["json"]["tbs"] == "qdr:d"
        assert kwargs["json"]["page"] == 1
        assert kwargs["headers"]["X-API-KEY"] == "test_key"


def test_serper_web_search_missing_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Serper API key not provided"):
            serper_web_search(query="test")


def test_serper_web_search_api_error():
    with patch("httpx.Client.post") as mock_post:
        mock_post.return_value = MagicMock(spec=httpx.Response)
        mock_post.return_value.status_code = 401
        mock_post.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized", request=MagicMock(), response=mock_post.return_value
        )

        with pytest.raises(httpx.HTTPStatusError):
            serper_web_search(query="test", api_key="invalid_key")


def test_serper_web_search_env_api_key():
    mock_response = {"searchParameters": {"q": "test"}, "organic": []}
    with patch.dict(os.environ, {"SERPER_API_KEY": "env_key"}):
        with patch("httpx.Client.post") as mock_post:
            mock_post.return_value = MagicMock(spec=httpx.Response)
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response

            serper_web_search(query="test")

            args, kwargs = mock_post.call_args
            assert kwargs["headers"]["X-API-KEY"] == "env_key"


@pytest.mark.skipif(
    not os.environ.get("SERPER_API_KEY"), reason="SERPER_API_KEY not provided"
)
def test_serper_web_search_integration():
    """Real API call test if key is available."""
    result = serper_web_search(query="kaval.ai")

    assert isinstance(result, SerperSearchResponse)
    assert result.organic is not None
    if result.organic:
        assert result.organic[0].link.startswith("http")
        assert result.organic[0].title
    for page in result.organic:
        print(page.title, page.link)
