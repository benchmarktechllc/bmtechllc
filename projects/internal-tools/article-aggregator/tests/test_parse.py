from __future__ import annotations

from datetime import UTC, datetime

import pytest

from article_aggregator.ingestion.parse import FeedParseError, parse_feed

from .conftest import fixture_bytes


def test_rss2() -> None:
    entries = parse_feed(fixture_bytes("rss2.xml"), "RSS Fixture")
    assert len(entries) == 5  # the item without title/link is dropped
    first = entries[0]
    assert first.title == "Hospital hit by ransomware attack"
    assert first.url == "https://news.example.com/2026/11/hospital-ransomware?utm_source=rss"
    assert first.guid == "example-1001"
    assert first.published_at == datetime(2026, 11, 18, 4, 0, tzinfo=UTC)
    assert "<b>ransomware</b>" in first.summary_html
    assert first.feed_name == "RSS Fixture"
    assert entries[3].published_at is None


def test_atom_uses_content_and_updated() -> None:
    entries = parse_feed(fixture_bytes("atom.xml"), "Atom Fixture")
    assert [e.title for e in entries] == [
        "Supply chain attack hits package registry",
        "Quarterly earnings roundup",
    ]
    assert entries[0].url == "https://atom.example.org/posts/supply-chain"
    assert entries[0].guid == "urn:example:atom:42"
    assert "supply chain attack" in entries[0].summary_html
    assert entries[0].published_at == datetime(2026, 11, 18, 3, 0, tzinfo=UTC)
    assert entries[1].summary_html == "Numbers went up."


def test_malformed_raises() -> None:
    with pytest.raises(FeedParseError, match="Broken Feed: not a readable feed"):
        parse_feed(fixture_bytes("malformed.xml"), "Broken Feed")


def test_empty_but_valid_feed_returns_nothing() -> None:
    xml = b'<?xml version="1.0"?><rss version="2.0"><channel><title>t</title></channel></rss>'
    assert parse_feed(xml, "Empty") == []
