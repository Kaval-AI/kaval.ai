from fastapi.testclient import TestClient
from kavalai.tools.rss import app, Feed
import kavalai.tools.rss as rss_module
from unittest.mock import MagicMock, patch
import pytest

client = TestClient(app)
auth = ("admin", "password")


@pytest.fixture(autouse=True)
def reset_auth():
    # Store original values
    orig_user = rss_module.AUTH_USER
    orig_password = rss_module.AUTH_PASSWORD
    yield
    # Restore original values after each test
    rss_module.AUTH_USER = orig_user
    rss_module.AUTH_PASSWORD = orig_password


def test_get_rss_feed_success():
    mock_feed = MagicMock()
    mock_feed.feed = {"title": "Test Feed"}
    mock_feed.entries = [
        {"title": "Entry 1", "link": "http://example.com/1", "summary": "Summary 1"},
        {"title": "Entry 2", "link": "http://example.com/2", "summary": "Summary 2"},
    ]

    with patch("feedparser.parse", return_value=mock_feed):
        response = client.get(
            "/get_rss_feed",
            params={"url": "http://example.com/rss"},
            auth=auth,
        )

    assert response.status_code == 200
    result = Feed(**response.json())
    assert result.title == "Test Feed"
    assert len(result.items) == 2
    assert result.items[0].title == "Entry 1"
    assert result.items[0].summary == "Summary 1"


def test_get_rss_feed_unauthorized():
    response = client.get(
        "/get_rss_feed",
        params={"url": "http://example.com/rss"},
        auth=("wrong", "pass"),
    )
    assert response.status_code == 401


def test_get_rss_feed_empty():
    mock_feed = MagicMock()
    mock_feed.entries = []

    with patch("feedparser.parse", return_value=mock_feed):
        response = client.get(
            "/get_rss_feed",
            params={"url": "http://example.com/rss"},
            auth=auth,
        )

    assert response.status_code == 200
    result = Feed(**response.json())
    assert result.title is None
    assert len(result.items) == 0


def test_get_rss_feed_max_results():
    mock_feed = MagicMock()
    mock_feed.feed = {"title": "Test Feed"}
    mock_feed.entries = [{"title": f"Entry {i}"} for i in range(10)]

    with patch("feedparser.parse", return_value=mock_feed):
        response = client.get(
            "/get_rss_feed",
            params={"url": "http://example.com/rss", "max_results": 3},
            auth=auth,
        )

    assert response.status_code == 200
    result = Feed(**response.json())
    assert len(result.items) == 3


def test_get_rss_feed_missing_fields():
    mock_feed = MagicMock()
    mock_feed.feed = {}
    mock_feed.entries = [{"other": "field"}]

    with patch("feedparser.parse", return_value=mock_feed):
        response = client.get(
            "/get_rss_feed",
            params={"url": "http://example.com/rss"},
            auth=auth,
        )

    assert response.status_code == 200
    result = Feed(**response.json())
    assert result.title is None
    assert result.items[0].title == "No Title"
    assert result.items[0].link == "No Link"
    assert result.items[0].summary == "No summary available."


def test_get_rss_feed_error():
    with patch("feedparser.parse", side_effect=Exception("Network error")):
        response = client.get(
            "/get_rss_feed",
            params={"url": "http://example.com/rss"},
            auth=auth,
        )
        assert response.status_code == 500
        assert "Error parsing feed: Network error" in response.json()["detail"]
