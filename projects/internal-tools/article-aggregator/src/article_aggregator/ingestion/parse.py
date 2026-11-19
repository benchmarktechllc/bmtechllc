"""Turn raw RSS/Atom bytes into Entry objects with feedparser."""

from __future__ import annotations

import calendar
from datetime import UTC, datetime
from typing import Any

import feedparser

from article_aggregator.ingestion.models import Entry
from article_aggregator.ingestion.text import normalize_text, strip_html


class FeedParseError(Exception):
    """The document could not be read as a feed at all."""


def parse_feed(content: bytes, feed_name: str) -> list[Entry]:
    parsed: Any = feedparser.parse(content)
    if parsed.get("bozo") and not parsed.get("entries"):
        reason = parsed.get("bozo_exception") or "no entries"
        raise FeedParseError(f"{feed_name}: not a readable feed ({reason})")
    entries: list[Entry] = []
    for raw in parsed.get("entries", []):
        entry = _to_entry(raw, feed_name)
        if entry is not None:
            entries.append(entry)
    return entries


def _to_entry(raw: Any, feed_name: str) -> Entry | None:
    title = normalize_text(strip_html(str(raw.get("title") or "")))
    url = str(raw.get("link") or "").strip()
    if not title or not url:
        return None
    guid = raw.get("id")
    return Entry(
        title=title,
        url=url,
        guid=str(guid).strip() or None if guid else None,
        summary_html=_summary_html(raw),
        published_at=_published_at(raw),
        feed_name=feed_name,
    )


def _summary_html(raw: Any) -> str:
    summary = raw.get("summary")
    if summary:
        return str(summary)
    for content in raw.get("content") or []:
        value = content.get("value")
        if value:
            return str(value)
    return ""


def _published_at(raw: Any) -> datetime | None:
    # feedparser exposes these as UTC struct_time values.
    parsed = raw.get("published_parsed") or raw.get("updated_parsed")
    if not parsed:
        return None
    return datetime.fromtimestamp(calendar.timegm(parsed), tz=UTC)
