from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Entry:
    """One feed item, normalized from feedparser output."""

    title: str
    url: str
    guid: str | None
    summary_html: str
    published_at: datetime | None
    feed_name: str


@dataclass(frozen=True)
class Match:
    """An entry plus the configured keywords it matched, in config order."""

    entry: Entry
    keywords: tuple[str, ...]
