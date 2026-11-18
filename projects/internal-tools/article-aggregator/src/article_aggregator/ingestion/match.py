"""Case-insensitive whole-word / whole-phrase keyword matching."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from article_aggregator.ingestion.models import Entry, Match
from article_aggregator.ingestion.text import normalize_text, strip_html


def _compile(keyword: str) -> re.Pattern[str]:
    words = normalize_text(keyword).split()
    body = r"\s+".join(re.escape(word) for word in words)
    # Lookarounds rather than \b so keywords that start or end with punctuation (C++) still work.
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


def searchable_text(entry: Entry) -> str:
    return normalize_text(f"{strip_html(entry.title)} {strip_html(entry.summary_html)}")


class KeywordMatcher:
    def __init__(self, keywords: Sequence[str]) -> None:
        self._patterns = [(keyword, _compile(keyword)) for keyword in keywords]

    def keywords_in(self, text: str) -> tuple[str, ...]:
        return tuple(keyword for keyword, pattern in self._patterns if pattern.search(text))

    def match(self, entry: Entry) -> Match | None:
        found = self.keywords_in(searchable_text(entry))
        return Match(entry=entry, keywords=found) if found else None

    def filter(self, entries: Iterable[Entry]) -> list[Match]:
        return [match for entry in entries if (match := self.match(entry)) is not None]
