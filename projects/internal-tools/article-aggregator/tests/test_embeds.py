from __future__ import annotations

from datetime import date, timedelta

from article_aggregator.discord.embeds import (
    EMBED_TOTAL_LIMIT,
    MAX_ARTICLES,
    SUMMARY_LIMIT,
    TITLE_LIMIT,
    build_digest_messages,
    build_embed,
    embed_char_count,
    escape_markdown,
    rank,
    select,
)

from .conftest import NOW, make_match

DAY = date(2026, 11, 18)
NO_MENTIONS = {"parse": []}


def titles(messages: list[dict[str, object]]) -> list[str]:
    return [m["embeds"][0]["title"] for m in messages[1:]]  # type: ignore[index]


def test_rank_by_keyword_count_then_newest_then_url() -> None:
    older_two = make_match(("a", "b"), title="older two", url="https://x/1",
                           published_at=NOW - timedelta(hours=5))
    newer_one = make_match(("a",), title="newer one", url="https://x/2", published_at=NOW)
    tie_b = make_match(("a",), title="tie b", url="https://x/b",
                       published_at=NOW - timedelta(hours=1))
    tie_a = make_match(("a",), title="tie a", url="https://x/a",
                       published_at=NOW - timedelta(hours=1))
    undated = make_match(("a",), title="undated", url="https://x/0", published_at=None)

    ranked = rank([undated, tie_b, newer_one, tie_a, older_two])

    assert [m.entry.title for m in ranked] == [
        "older two", "newer one", "tie a", "tie b", "undated"
    ]


def test_quiet_day_has_no_messages() -> None:
    assert build_digest_messages([], DAY) == []


def test_header_then_one_message_per_article() -> None:
    matches = [make_match(url=f"https://x/{i:02}", title=f"t{i:02}") for i in range(13)]
    messages = build_digest_messages(matches, DAY)

    header, *articles = messages
    assert header == {
        # Matches beyond the top 10 are not mentioned anywhere.
        "content": "**Digest 2026-11-18** · 10 articles",
        "allowed_mentions": NO_MENTIONS,
    }
    assert len(articles) == MAX_ARTICLES == 10
    for article in articles:
        assert set(article) == {"embeds", "allowed_mentions"}
        assert len(article["embeds"]) == 1
        assert article["allowed_mentions"] == NO_MENTIONS
    assert titles(messages) == [f"t{i:02}" for i in range(10)]


def test_no_more_line_when_everything_fits() -> None:
    header, article = build_digest_messages([make_match()], DAY)
    assert header["content"] == "**Digest 2026-11-18** · 1 article"
    assert article["embeds"][0]["title"] == "Title"


def test_feed_caps_skip_capped_feeds_and_backfill_from_others() -> None:
    # arXiv and TechCrunch rank highest on keyword count, but may take only 2 slots each.
    arxiv = [make_match(("a", "b", "c"), title=f"paper{i}", url=f"https://arxiv/{i}",
                        feed_name="arXiv") for i in range(6)]
    news = [make_match(("a", "b"), title=f"news{i}", url=f"https://news/{i}",
                       feed_name="TechCrunch") for i in range(5)]
    blogs = [make_match(("a",), title=f"blog{i}", url=f"https://blog/{i}",
                        feed_name="Blog") for i in range(8)]
    messages = build_digest_messages(arxiv + news + blogs, DAY, {"arXiv": 2, "TechCrunch": 2})
    assert titles(messages) == ["paper0", "paper1", "news0", "news1"] + [
        f"blog{i}" for i in range(6)
    ]
    assert messages[0]["content"] == "**Digest 2026-11-18** · 10 articles"


def test_feed_cap_can_leave_fewer_than_ten() -> None:
    arxiv = [make_match(url=f"https://arxiv/{i}", feed_name="arXiv") for i in range(6)]
    assert len(select(rank(arxiv), {"arXiv": 2})) == 2


def test_uncapped_feeds_are_unaffected() -> None:
    matches = [make_match(url=f"https://x/{i}", feed_name="Blog") for i in range(12)]
    assert len(select(rank(matches), {"arXiv": 2})) == MAX_ARTICLES


def test_embed_fields() -> None:
    match = make_match(
        ("ransomware", "zero day"),
        title="Hospital hit",
        url="https://news.example.com/a",
        summary_html="<p>Systems <b>encrypted</b>.</p>",
        feed_name="Example News",
    )
    embed = build_embed(match)
    assert embed["title"] == "Hospital hit"
    assert embed["url"] == "https://news.example.com/a"
    assert embed["description"] == "Systems encrypted."
    assert embed["footer"]["text"] == "Example News · matched: ransomware, zero day"
    assert embed["timestamp"] == "2026-11-18T06:00:00+00:00"


def test_untrusted_text_is_escaped_and_bad_urls_dropped() -> None:
    match = make_match(
        title="**Click** [here](https://evil.example) @everyone",
        url="javascript:alert(1)",
        summary_html="<p>__under__ `code` &lt;@123&gt;</p>",
    )
    embed = build_embed(match)
    assert embed["title"] == (
        r"\*\*Click\*\* \[here\]\(https\://evil.example\) \@everyone"
    )
    assert embed["description"] == r"\_\_under\_\_ \`code\` \<\@123\>"
    assert "url" not in embed


def test_summary_equal_to_title_is_dropped() -> None:
    embed = build_embed(make_match(title="Same text", summary_html="same TEXT"))
    assert "description" not in embed


def test_each_message_stays_within_discord_limits_for_worst_case_text() -> None:
    nasty = "*_" * 2000
    matches = [
        make_match(tuple(f"keyword{k}" for k in range(20)), title=nasty,
                   summary_html=f"<p>{'_`' * 2000}</p>", url=f"https://x/{i}",
                   feed_name="F" * 3000)
        for i in range(MAX_ARTICLES)
    ]
    for message in build_digest_messages(matches, DAY)[1:]:
        [embed] = message["embeds"]
        assert len(embed["title"]) <= TITLE_LIMIT
        assert len(embed["description"]) <= SUMMARY_LIMIT
        assert embed_char_count([embed]) <= EMBED_TOTAL_LIMIT


def test_escape_markdown_round_trip_characters() -> None:
    assert escape_markdown("a-b > c # d") == r"a\-b \> c \# d"
