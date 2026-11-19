from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from article_aggregator.ingestion.models import Entry, Match

FIXTURES = Path(__file__).parent / "fixtures"

# 09:00 EAT on 2026-11-18; the fixture feeds are dated relative to this.
NOW = datetime(2026, 11, 18, 6, 0, tzinfo=UTC)


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def make_entry(
    title: str = "Title",
    *,
    url: str = "https://example.com/a",
    summary_html: str = "",
    published_at: datetime | None = NOW,
    feed_name: str = "Feed",
) -> Entry:
    return Entry(
        title=title,
        url=url,
        guid=None,
        summary_html=summary_html,
        published_at=published_at,
        feed_name=feed_name,
    )


def make_match(keywords: tuple[str, ...] = ("kw",), **entry_kwargs: object) -> Match:
    return Match(entry=make_entry(**entry_kwargs), keywords=keywords)  # type: ignore[arg-type]


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    (tmp_path / "feeds.yaml").write_text(
        "- name: RSS Fixture\n  url: https://news.example.com/feed\n"
        "- name: Atom Fixture\n  url: https://atom.example.org/feed\n"
        "- name: Disabled\n  url: https://disabled.example.net/feed\n  enabled: false\n",
        encoding="utf-8",
    )
    (tmp_path / "keywords.yaml").write_text(
        "- ransomware\n- zero day\n- data breach\n- supply chain attack\n", encoding="utf-8"
    )
    return tmp_path
