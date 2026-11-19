"""Slice 1 digest: fetch, parse, keep the last 24 h, match, then post a header message
followed by one message per article.

No database yet, so a second run on the same day posts again, and a run that fails partway
leaves the messages it already posted in the channel.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from article_aggregator.config import AppConfig
from article_aggregator.discord.embeds import build_digest_messages
from article_aggregator.ingestion.fetch import fetch_feeds
from article_aggregator.ingestion.match import KeywordMatcher
from article_aggregator.ingestion.models import Entry, Match
from article_aggregator.ingestion.parse import FeedParseError, parse_feed

log = logging.getLogger(__name__)

DIGEST_TZ = ZoneInfo("Africa/Nairobi")
LOOKBACK = timedelta(hours=24)

Poster = Callable[[dict[str, Any]], str]


class AllFeedsFailedError(Exception):
    pass


class PartialPostError(Exception):
    """Posting stopped partway; some digest messages are already in the channel."""

    def __init__(self, posted: int, total: int, cause: Exception) -> None:
        super().__init__(f"posted {posted} of {total} digest messages, then failed: {cause}")
        self.posted = posted
        self.total = total


@dataclass(frozen=True)
class DigestResult:
    digest_date: date
    feeds_ok: int
    feeds_failed: int
    entries_seen: int
    recent: int
    matched: int
    payloads: list[dict[str, Any]]
    message_ids: list[str]


def digest_date_for(now: datetime) -> date:
    return now.astimezone(DIGEST_TZ).date()


def select_recent(entries: Iterable[Entry], now: datetime) -> list[Entry]:
    """Entries published in the last 24 h. Undated entries are skipped in slice 1."""
    cutoff = now - LOOKBACK
    return [e for e in entries if e.published_at is not None and e.published_at >= cutoff]


def unique_by_url(matches: Iterable[Match]) -> list[Match]:
    """Drop exact-URL repeats within one run (the same item listed in two feeds)."""
    seen: set[str] = set()
    unique: list[Match] = []
    for match in matches:
        if match.entry.url not in seen:
            seen.add(match.entry.url)
            unique.append(match)
    return unique


def run_digest(
    config: AppConfig,
    *,
    now: datetime,
    poster: Poster | None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> DigestResult:
    """Build today's digest and post it with `poster`; `poster=None` is a dry run."""
    feeds = config.enabled_feeds
    results = asyncio.run(fetch_feeds(feeds, transport=transport))

    entries: list[Entry] = []
    failed = 0
    for fetched in results:
        if fetched.content is None:
            failed += 1
            continue
        try:
            entries.extend(parse_feed(fetched.content, fetched.feed.name))
        except FeedParseError as exc:
            failed += 1
            log.warning("%s", exc)
    if failed == len(feeds):
        raise AllFeedsFailedError(f"all {failed} enabled feeds failed; nothing was posted")

    recent = select_recent(entries, now)
    matches = unique_by_url(KeywordMatcher(config.keywords).filter(recent))
    digest_date = digest_date_for(now)
    payloads = build_digest_messages(matches, digest_date, config.feed_caps)

    message_ids: list[str] = []
    if poster is not None:
        for payload in payloads:
            try:
                message_ids.append(poster(payload))
            except Exception as exc:
                raise PartialPostError(len(message_ids), len(payloads), exc) from exc

    result = DigestResult(
        digest_date=digest_date,
        feeds_ok=len(feeds) - failed,
        feeds_failed=failed,
        entries_seen=len(entries),
        recent=len(recent),
        matched=len(matches),
        payloads=payloads,
        message_ids=message_ids,
    )
    log.info(
        "digest %s: feeds ok=%d failed=%d, entries=%d, last24h=%d, matched=%d, messages=%d",
        digest_date,
        result.feeds_ok,
        result.feeds_failed,
        result.entries_seen,
        result.recent,
        result.matched,
        len(message_ids),
    )
    return result
