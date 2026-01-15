import pytest
from kavalai.tools.rss import get_rss_feed, Feed
from unittest.mock import MagicMock, patch


def test_get_rss_feed_success():
    mock_feed = MagicMock()
    mock_feed.feed = {"title": "Test Feed"}
    mock_feed.entries = [
        {"title": "Entry 1", "link": "http://example.com/1", "summary": "Summary 1"},
        {"title": "Entry 2", "link": "http://example.com/2", "summary": "Summary 2"},
    ]

    with patch("feedparser.parse", return_value=mock_feed):
        result = get_rss_feed("http://example.com/rss")

    assert isinstance(result, Feed)
    assert result.title == "Test Feed"
    assert len(result.items) == 2
    assert result.items[0].title == "Entry 1"
    assert result.items[0].summary == "Summary 1"


def test_get_rss_feed_empty():
    mock_feed = MagicMock()
    mock_feed.entries = []

    with patch("feedparser.parse", return_value=mock_feed):
        result = get_rss_feed("http://example.com/rss")

    assert isinstance(result, Feed)
    assert result.title is None
    assert len(result.items) == 0


def test_get_rss_feed_max_results():
    mock_feed = MagicMock()
    mock_feed.feed = {"title": "Test Feed"}
    mock_feed.entries = [{"title": f"Entry {i}"} for i in range(10)]

    with patch("feedparser.parse", return_value=mock_feed):
        result = get_rss_feed("http://example.com/rss", max_results=3)

    assert len(result.items) == 3


def test_get_rss_feed_missing_fields():
    mock_feed = MagicMock()
    mock_feed.feed = {}
    mock_feed.entries = [{"other": "field"}]

    with patch("feedparser.parse", return_value=mock_feed):
        result = get_rss_feed("http://example.com/rss")

    assert result.title is None
    assert result.items[0].title == "No Title"
    assert result.items[0].link == "No Link"
    assert result.items[0].summary == "No summary available."


def test_get_rss_feed_error():
    with patch("feedparser.parse", side_effect=Exception("Network error")):
        with pytest.raises(Exception, match="Error parsing feed: Network error"):
            get_rss_feed("http://example.com/rss")
