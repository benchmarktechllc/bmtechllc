"""Rank matches and build the digest messages: one header, then one message per article.

Feed text is untrusted: everything that Discord renders as markdown is escaped, and every
payload disables all mentions.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any
from urllib.parse import urlsplit

from article_aggregator.ingestion.models import Match
from article_aggregator.ingestion.text import strip_html, truncate

MAX_ARTICLES = 10
TITLE_LIMIT = 256
SUMMARY_LIMIT = 300
FOOTER_LIMIT = 2048
CONTENT_LIMIT = 2000
EMBED_TOTAL_LIMIT = 6000
EMBED_COLOR = 0x2F6FDE
NO_MENTIONS: dict[str, list[str]] = {"parse": []}

_MARKDOWN = re.compile(r"([\\*_~`|>#\[\]()<@:-])")
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

Payload = dict[str, Any]


def rank(matches: Sequence[Match]) -> list[Match]:
    """Most distinct keywords first, then newest, then URL for a stable order."""
    return sorted(
        matches,
        key=lambda m: (
            -len(set(m.keywords)),
            -(m.entry.published_at or _EPOCH).timestamp(),
            m.entry.url,
        ),
    )


def select(
    ranked: Sequence[Match], feed_caps: Mapping[str, int], limit: int = MAX_ARTICLES
) -> list[Match]:
    """Take up to `limit` matches in rank order, skipping any whose feed has hit its cap."""
    taken: dict[str, int] = {}
    chosen: list[Match] = []
    for match in ranked:
        if len(chosen) == limit:
            break
        feed = match.entry.feed_name
        cap = feed_caps.get(feed)
        if cap is not None and taken.get(feed, 0) >= cap:
            continue
        taken[feed] = taken.get(feed, 0) + 1
        chosen.append(match)
    return chosen


def escape_markdown(text: str) -> str:
    return _MARKDOWN.sub(r"\\\1", text)


def _fit(text: str, limit: int) -> str:
    """Truncate then escape, shrinking until the escaped text fits in `limit`."""
    budget = limit
    while True:
        escaped = escape_markdown(truncate(text, budget))
        if len(escaped) <= limit:
            return escaped
        budget -= max(1, len(escaped) - limit)


def _safe_url(url: str) -> str | None:
    return url if urlsplit(url).scheme in ("http", "https") else None


def build_embed(match: Match) -> Payload:
    entry = match.entry
    embed: Payload = {
        "title": _fit(entry.title, TITLE_LIMIT),
        "color": EMBED_COLOR,
        "footer": {
            "text": truncate(f"{entry.feed_name} · matched: {', '.join(match.keywords)}",
                             FOOTER_LIMIT),
        },
    }
    url = _safe_url(entry.url)
    if url:
        embed["url"] = url
    summary = strip_html(entry.summary_html)
    if summary and summary.casefold() != entry.title.casefold():
        embed["description"] = _fit(summary, SUMMARY_LIMIT)
    if entry.published_at:
        embed["timestamp"] = entry.published_at.astimezone(UTC).isoformat()
    return embed


def embed_char_count(embeds: Sequence[Payload]) -> int:
    """Characters Discord counts toward the 6,000 per-message embed limit."""
    total = 0
    for embed in embeds:
        total += len(embed.get("title", "")) + len(embed.get("description", ""))
        total += len(embed.get("footer", {}).get("text", ""))
        total += len(embed.get("author", {}).get("name", ""))
        for field in embed.get("fields", []):
            total += len(field.get("name", "")) + len(field.get("value", ""))
    return total


def build_digest_messages(
    matches: Sequence[Match], digest_date: date, feed_caps: Mapping[str, int] | None = None
) -> list[Payload]:
    """Header message, then one single-embed message per article. Empty on a quiet day."""
    if not matches:
        return []
    ranked = rank(matches)
    shown = select(ranked, feed_caps or {})
    noun = "article" if len(shown) == 1 else "articles"
    content = f"**Digest {digest_date.isoformat()}** · {len(shown)} {noun}"
    header: Payload = {"content": content[:CONTENT_LIMIT], "allowed_mentions": NO_MENTIONS}
    articles: list[Payload] = [
        {"embeds": [build_embed(match)], "allowed_mentions": NO_MENTIONS} for match in shown
    ]
    return [header, *articles]
