"""Meeting lane: detector, date resolution, evidence order, labels, and citations. Synthetic data only."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import Any

import pytest

from benchmark_cli import agent, discord_ask, meeting_answers
from benchmark_cli.meeting_answers import (
    Meeting,
    asks_about_meeting,
    meeting_label,
    meeting_not_found_response,
    parse_requested_date,
    resolve_meeting,
    today_in_new_york,
)
from benchmark_cli.models import Citation
from benchmark_cli.query_understanding import QueryPlan
from benchmark_cli.retrieval import HybridSearchResult, RetrievalDecision
from benchmark_cli.storage.postgres import SearchChunk

TODAY = date(2026, 11, 13)  # a Friday
SYNC = Meeting("fireflies/aaa.md", "Team Sync Call", "2026-11-13", "https://example.test/aaa")
OLDER_SYNC = Meeting("fireflies/bbb.md", "Team Sync Call", "2026-11-06", "https://example.test/bbb")
PLANNING = Meeting("fathom/ccc.md", "Quarterly Planning", "2026-11-10", "")
MEETINGS = [OLDER_SYNC, PLANNING, SYNC]


@pytest.mark.unit
@pytest.mark.parametrize(
    "question",
    [
        "what were the todos from the november 13 sync call?",
        "what did we discuss in the last Sync Call",
        "summarize the meeting",
        "who owns automating meeting notes to discord?",
        "what was said in the standup",
        "show me the transcript",
        "what did we agree in our call yesterday",
        "weekly call agenda",
    ],
)
def test_detector_positives(question: str) -> None:
    assert asks_about_meeting(question)


@pytest.mark.unit
@pytest.mark.parametrize(
    "question",
    [
        "where is Discord message sync",
        "what did we decide about Fireflies?",
        "how many people work at bmtech",
        "call the vendor",
        "does the sync job run nightly",
        "tell me about Fathom",
        "",
    ],
)
def test_detector_negatives(question: str) -> None:
    assert not asks_about_meeting(question)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("todos from the november 13 sync call", date(2026, 11, 13)),
        ("nov 13th meeting", date(2026, 11, 13)),
        ("13 november meeting", date(2026, 11, 13)),
        ("meeting on 11/13", date(2026, 11, 13)),
        ("meeting on 11/13/2025", date(2025, 11, 13)),
        ("meeting on 2026-11-13", date(2026, 11, 13)),
        ("meeting on december 2", date(2025, 12, 2)),  # yearless and after today: last year
        ("meeting on october 2", date(2026, 10, 2)),
        ("meeting on november 14, 2027", date(2027, 11, 14)),
        ("today's meeting", TODAY),
        ("yesterday's call", date(2026, 11, 12)),
        ("the last friday meeting", date(2026, 11, 6)),
        ("the last monday meeting", date(2026, 11, 9)),
        ("meeting on february 30", None),
        ("what happened in the last sync call", None),
    ],
)
def test_parse_requested_date(question: str, expected: date | None) -> None:
    assert parse_requested_date(question, TODAY) == expected


@pytest.mark.unit
def test_leap_day_yearless_uses_most_recent_leap_year() -> None:
    assert parse_requested_date("meeting on feb 29", TODAY) == date(2024, 2, 29)


@pytest.mark.unit
def test_today_is_the_new_york_date() -> None:
    late = datetime(2026, 11, 14, 2, 30, tzinfo=UTC)  # 21:30 on the 13th in New York
    assert today_in_new_york(late) == date(2026, 11, 13)


@pytest.mark.unit
def test_resolve_explicit_date_and_utc_rollover() -> None:
    assert resolve_meeting("todos from the november 13 sync call", MEETINGS, TODAY).meeting == SYNC
    evening = Meeting("fireflies/ddd.md", "Evening Review", "2026-11-11", "")
    found = resolve_meeting("notes from the meeting on november 10", [evening, SYNC], TODAY)
    assert found.meeting == evening  # stored one day later than it happened locally
    exact_first = resolve_meeting("meeting on november 10", [evening, PLANNING], TODAY)
    assert exact_first.meeting == PLANNING


@pytest.mark.unit
def test_resolve_recency_prefers_title_overlap_then_newest() -> None:
    assert resolve_meeting("what did we discuss in the last sync call", MEETINGS, TODAY).meeting == SYNC
    assert resolve_meeting("the previous planning meeting", MEETINGS, TODAY).meeting == PLANNING
    assert resolve_meeting("the latest meeting", MEETINGS, TODAY).meeting == SYNC
    assert resolve_meeting("most recent standup", [OLDER_SYNC, PLANNING], TODAY).meeting == PLANNING


@pytest.mark.unit
def test_resolve_without_date_or_recency_picks_nothing() -> None:
    resolution = resolve_meeting("who owns automating meeting notes", MEETINGS, TODAY)
    assert resolution.meeting is None
    assert not resolution.not_found


@pytest.mark.unit
def test_resolve_named_date_with_no_meeting_is_not_found() -> None:
    resolution = resolve_meeting("todos from the october 2 sync call", MEETINGS, TODAY)
    assert resolution.meeting is None
    assert resolution.not_found
    assert resolution.requested == date(2026, 10, 2)


@pytest.mark.unit
def test_not_found_reply_lists_indexed_meetings_without_citations() -> None:
    reply = meeting_not_found_response(date(2026, 10, 2), MEETINGS)
    assert reply.status == "no_relevant_information"
    assert reply.citations == []
    assert reply.confidence == 0.0
    assert reply.answer.startswith("I have no meeting indexed for October 2, 2026.")
    assert "Team Sync Call (2026-11-13)" in reply.answer
    assert reply.answer.index("2026-11-13") < reply.answer.index("2026-11-06")
    assert "No meetings are indexed" in meeting_not_found_response(date(2026, 10, 2), []).answer


@pytest.mark.unit
def test_label_needs_title_and_date() -> None:
    assert meeting_label({"title": 'Team "Sync"', "date": "2026-11-13", "section": "Summary"}) == (
        "Meeting \"Team 'Sync'\" on 2026-11-13, Summary"
    )
    assert meeting_label({"title": "Team Sync", "date": ""}) is None


def _chunk(chunk_id: int, source: str, section: str, text: str = "text") -> SearchChunk:
    metadata = {
        "source_type": "transcript",
        "kind": "meeting_transcript",
        "title": "Team Sync Call",
        "date": "2026-11-13",
        "section": section,
    }
    return SearchChunk(chunk_id, source, 0, text, metadata, 0.5, content_sha256="a" * 64)


@pytest.mark.unit
def test_payload_label_unchanged_outside_the_lane() -> None:
    chunk = _chunk(1, "fireflies/aaa.md", "Summary")
    assert agent._chunk_payload(chunk)["source_label"] == "Meeting transcript"
    lane = agent._chunk_payload(chunk, meeting_lane=True)
    assert lane["source_label"] == 'Meeting "Team Sync Call" on 2026-11-13, Summary'
    plain = agent._chunk_payload(chunk)
    assert {k: v for k, v in lane.items() if k not in {"source_label", "text"}} == {
        k: v for k, v in plain.items() if k not in {"source_label", "text"}
    }
    untitled = SearchChunk(2, "x.md", 0, "t", {"source_type": "transcript"}, 0.1, content_sha256="a" * 64)
    assert agent._chunk_payload(untitled, meeting_lane=True)["source_label"] == "Meeting transcript"


def _citation(source: str) -> Citation:
    return Citation(source_file=source, chunk_id=7, content_sha256="b" * 64, quote="a quoted line")


@pytest.mark.unit
def test_citation_shows_title_date_and_url_for_meeting_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(meeting_answers, "list_meetings", lambda source_file=None: [(SYNC.source_file, SYNC.title, SYNC.date, SYNC.url)])
    line = discord_ask._format_citation(1, _citation("fireflies/aaa.md"))
    assert line == "1. Team Sync Call (2026-11-13) <https://example.test/aaa> (chunk 7): a quoted line"


@pytest.mark.unit
def test_citation_unchanged_for_other_sources_and_on_lookup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(source_file: str | None = None) -> list[tuple[str, str, str, str]]:
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(meeting_answers, "list_meetings", boom)
    assert discord_ask._format_citation(1, _citation("kb/page.md")) == "1. `kb/page.md` (chunk 7): a quoted line"
    assert discord_ask._format_citation(2, _citation("fireflies/aaa.md")) == "2. `fireflies/aaa.md` (chunk 7): a quoted line"
    called: list[str] = []
    monkeypatch.setattr(meeting_answers, "list_meetings", lambda source_file=None: called.append("x") or [])
    discord_ask._format_citation(1, _citation("trello/card.md"))
    assert called == []


# --------------------------------------------------------------------------- run_agent


def _install(monkeypatch: pytest.MonkeyPatch, question: str, normal: list[SearchChunk]) -> dict[str, Any]:
    seen: dict[str, Any] = {"retrievals": [], "messages": None, "meeting_lists": 0}
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **kwargs: seen.update(kwargs))
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [])
    monkeypatch.setattr(agent, "understand_query", lambda *_a, **_k: QueryPlan(searches=[question], person=None))
    monkeypatch.setattr(agent, "fts_search", lambda *_a, **_k: [])
    monkeypatch.setattr(agent, "today_in_new_york", lambda: TODAY)

    def indexed(source_file: str | None = None) -> list[Meeting]:
        seen["meeting_lists"] += 1
        return list(MEETINGS)

    monkeypatch.setattr(agent, "indexed_meetings", indexed)
    monkeypatch.setattr(
        agent,
        "meeting_summary_chunks",
        lambda source: [_chunk(1, source, "Overview"), _chunk(2, source, "Summary")],
    )

    def retrieve(query: str, **kwargs: Any) -> RetrievalDecision:
        seen["retrievals"].append(kwargs.get("source"))
        if kwargs.get("source"):
            hits = [_chunk(10 + i, kwargs["source"], "Transcript") for i in range(9)]
        else:
            hits = normal
        return RetrievalDecision(
            "accepted", tuple(HybridSearchResult(c, 0.1, 0.8, 1.0) for c in hits), 0.8, 3
        )

    monkeypatch.setattr(agent, "retrieve_evidence", retrieve)

    def call_llm(**kwargs: Any) -> Any:
        seen["messages"] = kwargs["messages"]
        raise RuntimeError("stop here")

    monkeypatch.setattr(agent, "call_llm", call_llm)
    monkeypatch.setattr(
        agent, "call_llm_structured", lambda **_k: (_ for _ in ()).throw(RuntimeError("stop here"))
    )
    return seen


def _retrieved(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence = next(m for m in messages if m["content"].startswith("Initial evidence"))
    return json.loads(evidence["content"].split("\n", 1)[1])["retrieved"]


@pytest.mark.unit
def test_lane_puts_the_resolved_meeting_first(monkeypatch: pytest.MonkeyPatch) -> None:
    question = "what were the todos from the november 13 sync call?"
    other = _chunk(50, "fireflies/zzz.md", "Transcript")
    same_meeting = _chunk(51, "fireflies/aaa.md", "Transcript")
    unrelated = SearchChunk(60, "kb/page.md", 0, "x", {"source_type": "kb_page"}, 0.4, content_sha256="c" * 64)
    seen = _install(monkeypatch, question, [same_meeting, other, unrelated])
    agent.run_agent(question)
    retrieved = _retrieved(seen["messages"])
    ids = [item["chunk_id"] for item in retrieved]
    assert ids[:2] == [1, 2]
    assert ids[2:8] == [10, 11, 12, 13, 14, 15]
    assert ids[8:] == [50, 60]  # normal hits minus the resolved meeting's own chunks
    assert retrieved[1]["source_label"] == 'Meeting "Team Sync Call" on 2026-11-13, Summary'
    assert retrieved[-1]["source_label"] == "Knowledge base"
    assert seen["retrievals"] == [None, "fireflies/aaa.md"]
    lane_messages = [m for m in seen["messages"] if m["content"] == meeting_answers.MEETING_LANE_PROMPT]
    assert len(lane_messages) == 1 and lane_messages[0]["role"] == "system"


@pytest.mark.unit
def test_lane_caps_evidence_at_twenty(monkeypatch: pytest.MonkeyPatch) -> None:
    question = "todos from the last sync call"
    normal = [
        SearchChunk(100 + i, f"kb/{i}.md", 0, "x", {"source_type": "kb_page"}, 0.4, content_sha256="c" * 64)
        for i in range(30)
    ]
    seen = _install(monkeypatch, question, normal)
    agent.run_agent(question)
    assert len(_retrieved(seen["messages"])) == 20


@pytest.mark.unit
def test_missing_date_returns_meeting_not_found_without_a_model_call(monkeypatch: pytest.MonkeyPatch) -> None:
    question = "what were the todos from the october 2 sync call?"
    seen = _install(monkeypatch, question, [])
    response = agent.run_agent(question)
    assert seen["outcome"] == "meeting_not_found"
    assert seen["messages"] is None
    assert response.citations == []
    assert "Team Sync Call (2026-11-13)" in response.answer


@pytest.mark.unit
def test_meeting_question_without_a_meeting_only_gets_labels_and_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    question = "who owns automating meeting notes to discord?"
    hit = _chunk(5, "fireflies/aaa.md", "Summary")
    seen = _install(monkeypatch, question, [hit])
    agent.run_agent(question)
    assert seen["retrievals"] == [None]
    assert _retrieved(seen["messages"])[0]["source_label"].startswith("Meeting \"Team Sync Call\"")
    assert any(m["content"] == meeting_answers.MEETING_LANE_PROMPT for m in seen["messages"])


@pytest.mark.unit
@pytest.mark.parametrize("question", ["what did we decide about Fireflies?", "where is Discord message sync"])
def test_non_meeting_questions_keep_todays_path(monkeypatch: pytest.MonkeyPatch, question: str) -> None:
    hit = _chunk(5, "fireflies/aaa.md", "Summary")
    seen = _install(monkeypatch, question, [hit])
    agent.run_agent(question)
    assert seen["meeting_lists"] == 0
    assert seen["retrievals"] == [None]
    assert [item["source_label"] for item in _retrieved(seen["messages"])] == ["Meeting transcript"]
    assert not any(m["content"] == meeting_answers.MEETING_LANE_PROMPT for m in seen["messages"])
    assert [m["role"] for m in seen["messages"]] == ["system", "system", "user"]


