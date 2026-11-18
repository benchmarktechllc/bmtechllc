from __future__ import annotations

import pytest

from article_aggregator.ingestion.match import KeywordMatcher
from article_aggregator.ingestion.text import strip_html, truncate

from .conftest import make_entry


@pytest.mark.parametrize(
    ("keyword", "text", "expected"),
    [
        ("AI", "Analysts said the deal is off", False),
        ("AI", "New AI rules published", True),
        ("AI", "ai-powered phishing", True),
        ("AI", "AIs are everywhere", False),
        ("zero day", "Vendor patches Zero  Day flaw", True),
        ("zero day", "Vendor patches Zero\n\tDay flaw", True),
        ("zero day", "a zero-day flaw", False),
        ("C++", "Why C++ still matters", True),
        ("C++", "C++20 modules", False),
        ("C", "Use C++ instead", True),
        ("café", "CAFÉ OWNERS SUE", True),
        ("café", "café owners sue", True),
        ("AI", "ＡＩ fullwidth headline", True),
        ("ransomware", "Ransomware gang returns", True),
        ("ransomware", "antiransomware tool", False),
        # Behaviour of the production keyword list, pinned so changes are deliberate.
        ("agent", "Building AI agents in production", False),
        ("agent", "A multi-agent planner", True),
        ("multi-agent", "Multi-Agent systems at scale", True),
        ("multi-agent", "multi agent systems", False),
        ("llama_index", "Using llama_index with Postgres", True),
        ("llama_index", "Using LlamaIndex with Postgres", False),
        ("mcp", "An MCP server for Jira", True),
    ],
)
def test_whole_word_case_insensitive(keyword: str, text: str, expected: bool) -> None:
    matcher = KeywordMatcher([keyword])
    assert bool(matcher.match(make_entry(text))) is expected


def test_matches_summary_html_and_reports_keywords_in_config_order() -> None:
    matcher = KeywordMatcher(["zero day", "ransomware", "breach"])
    entry = make_entry(
        "Weekly update",
        summary_html="<p>A <b>ransomware</b> crew used a <i>zero</i> <i>day</i>.</p>",
    )
    match = matcher.match(entry)
    assert match is not None
    assert match.keywords == ("zero day", "ransomware")


def test_html_tags_split_words_and_script_is_ignored() -> None:
    matcher = KeywordMatcher(["ransomware"])
    assert matcher.match(make_entry("x", summary_html="<p>ransom</p><p>ware</p>")) is None
    assert matcher.match(make_entry("x", summary_html="<script>ransomware</script>")) is None


def test_html_entities_decoded() -> None:
    assert strip_html("Tom &amp; Jerry&nbsp;&lt;3") == "Tom & Jerry <3"


def test_filter_keeps_only_matches() -> None:
    matcher = KeywordMatcher(["breach"])
    matches = matcher.filter([make_entry("Data breach"), make_entry("Nothing")])
    assert [m.entry.title for m in matches] == ["Data breach"]


@pytest.mark.parametrize(
    ("text", "limit", "expected"),
    [
        ("short", 10, "short"),
        ("one two three four", 12, "one two…"),
        ("unbreakablewordhere", 8, "unbreak…"),
        ("trailing, comma here", 11, "trailing…"),
    ],
)
def test_truncate(text: str, limit: int, expected: str) -> None:
    result = truncate(text, limit)
    assert result == expected
    assert len(result) <= limit
