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
from kavalai.tools.websearch.google_custom_search import (
    google_custom_search,
    CustomSearchResponse,
)


def test_google_custom_search_success():
    mock_response = {
        "items": [
            {
                "title": "Test Result",
                "link": "https://example.com",
                "snippet": "This is a test snippet.",
                "displayLink": "example.com",
            }
        ],
        "searchInformation": {"totalResults": "1", "searchTime": 0.123},
    }

    with patch("httpx.Client.get") as mock_get:
        mock_get.return_value = MagicMock(spec=httpx.Response)
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status = MagicMock()

        result = google_custom_search(
            query="test query", api_key="test_key", cx="test_cx", num=5
        )

        assert isinstance(result, CustomSearchResponse)
        assert len(result.items) == 1
        assert result.items[0].title == "Test Result"
        assert result.searchInformation.totalResults == "1"

        # Verify request parameters
        args, kwargs = mock_get.call_args
        assert kwargs["params"]["q"] == "test query"
        assert kwargs["params"]["key"] == "test_key"
        assert kwargs["params"]["cx"] == "test_cx"
        assert kwargs["params"]["num"] == 5


def test_google_custom_search_missing_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Google API key not provided"):
            google_custom_search(query="test", cx="test_cx")


def test_google_custom_search_missing_cx():
    with patch.dict(
        os.environ, {"GOOGLE_CUSTOM_SEARCH_API_KEY": "test_key"}, clear=True
    ):
        with pytest.raises(
            ValueError, match=r"Programmable Search Engine ID \(cx\) not provided"
        ):
            google_custom_search(query="test")


def test_google_custom_search_api_error():
    with patch("httpx.Client.get") as mock_get:
        mock_get.return_value = MagicMock(spec=httpx.Response)
        mock_get.return_value.status_code = 403
        mock_get.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Forbidden", request=MagicMock(), response=mock_get.return_value
        )

        with pytest.raises(httpx.HTTPStatusError):
            google_custom_search(query="test", api_key="invalid_key", cx="invalid_cx")


def test_google_custom_search_env_vars():
    mock_response = {"items": []}
    with patch.dict(
        os.environ,
        {
            "GOOGLE_CUSTOM_SEARCH_API_KEY": "env_key",
            "GOOGLE_CUSTOM_SEARCH_CX": "env_cx",
        },
    ):
        with patch("httpx.Client.get") as mock_get:
            mock_get.return_value = MagicMock(spec=httpx.Response)
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_response

            google_custom_search(query="test")

            args, kwargs = mock_get.call_args
            assert kwargs["params"]["key"] == "env_key"
            assert kwargs["params"]["cx"] == "env_cx"


@pytest.mark.skipif(
    not (
        os.environ.get("GOOGLE_CUSTOM_SEARCH_API_KEY")
        and os.environ.get("GOOGLE_CUSTOM_SEARCH_CX")
    ),
    reason="GOOGLE_CUSTOM_SEARCH_API_KEY or GOOGLE_CUSTOM_SEARCH_CX not provided",
)
def test_google_custom_search_integration():
    """Real API call test if key and cx are available."""
    result = google_custom_search(query="kaval.ai", num=1)

    assert isinstance(result, CustomSearchResponse)
    if result.items:
        assert result.items[0].link.startswith("http")
        assert result.items[0].title
    for item in result.items:
        print(item.title, item.link)
