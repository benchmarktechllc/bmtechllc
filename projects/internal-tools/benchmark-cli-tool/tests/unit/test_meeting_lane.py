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
def test_payload_label_names_the_meeting_when_title_and_date_exist() -> None:
    chunk = _chunk(1, "fireflies/aaa.md", "Summary")
    payload = agent._chunk_payload(chunk)
    assert payload["source_label"] == 'Meeting "Team Sync Call" on 2026-11-13, Summary'
    untitled = SearchChunk(2, "x.md", 0, "t", {"source_type": "transcript"}, 0.1, content_sha256="a" * 64)
    assert agent._chunk_payload(untitled)["source_label"] == "Meeting transcript"


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

CARD = ("board", "Doing", "c1", "Task alpha", "https://example.test/c1", 900, "trello/c1.md", "e" * 64, "Task alpha card")


def _meeting_chunks(source: str, count: int = 6) -> list[SearchChunk]:
    sections = ["Overview", "Summary"] + ["Transcript"] * (count - 2)
    return [
        SearchChunk(
            index + 1,
            source,
            index,
            f"Synthetic line {index + 1}",
            {
                "source_type": "transcript",
                "kind": "meeting_transcript",
                "title": "Team Sync Call",
                "date": "2026-11-13",
                "section": section,
            },
            0.0,
            content_sha256="a" * 64,
        )
        for index, section in enumerate(sections)
    ]


def _install(
    monkeypatch: pytest.MonkeyPatch,
    question: str,
    normal: list[SearchChunk],
    drafts: list[Any] | None = None,
    *,
    cards: bool = False,
) -> dict[str, Any]:
    seen: dict[str, Any] = {"retrievals": [], "messages": None, "meeting_lists": 0, "loaded": [], "calls": []}
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **kwargs: seen.update(kwargs))
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [CARD] if cards else [])
    monkeypatch.setattr(agent, "understand_query", lambda *_a, **_k: QueryPlan(searches=[question], person=None))
    monkeypatch.setattr(agent, "fts_search", lambda *_a, **_k: [])
    monkeypatch.setattr(agent, "today_in_new_york", lambda *_a: TODAY)
    monkeypatch.setattr(agent, "classify_query", lambda q: None)

    def indexed(source_file: str | None = None) -> list[Meeting]:
        seen["meeting_lists"] += 1
        return list(MEETINGS)

    monkeypatch.setattr(agent, "indexed_meetings", indexed)

    def load(source: str) -> list[SearchChunk]:
        seen["loaded"].append(source)
        return _meeting_chunks(source)

    monkeypatch.setattr(agent, "meeting_chunks", load)

    def retrieve(query: str, **kwargs: Any) -> RetrievalDecision:
        seen["retrievals"].append(kwargs.get("source"))
        hits = _meeting_chunks(kwargs["source"])[2:] if kwargs.get("source") else normal
        return RetrievalDecision(
            "accepted" if hits else "no_relevant_information",
            tuple(HybridSearchResult(c, 0.1, 0.8, 1.0) for c in hits),
            0.8,
            3,
        )

    monkeypatch.setattr(agent, "retrieve_evidence", retrieve)

    def call_llm(**kwargs: Any) -> Any:
        seen["tool_loop"] = True
        seen["messages"] = kwargs["messages"]
        return type("C", (), {"choices": [type("M", (), {"message": type("X", (), {"tool_calls": None, "content": ""})()})()]})()

    def synth(*, tier: str, messages: list[dict[str, Any]]) -> Any:
        seen["calls"].append(messages)
        if seen["messages"] is None:
            seen["messages"] = messages
        index = len(seen["calls"]) - 1
        if not drafts:
            return None
        return drafts[min(index, len(drafts) - 1)]

    monkeypatch.setattr(agent, "call_llm", call_llm)
    monkeypatch.setattr(agent, "_synthesize_answer", synth)
    return seen


def _pack(messages: list[dict[str, Any]]) -> dict[str, Any]:
    evidence = next(m for m in messages if m["content"].startswith("Initial evidence"))
    return json.loads(evidence["content"].split("\n", 1)[1])


@pytest.mark.unit
def test_full_meeting_is_in_the_pack_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    question = "what were the todos from the november 13 sync call?"
    other = _chunk(50, "fireflies/zzz.md", "Transcript")
    same_meeting = _chunk(3, "fireflies/aaa.md", "Transcript")
    unrelated = SearchChunk(60, "kb/page.md", 0, "x", {"source_type": "kb_page"}, 0.4, content_sha256="c" * 64)
    seen = _install(monkeypatch, question, [same_meeting, other, unrelated], [_resp("Synthetic line 1", "x", 1)], cards=True)
    agent.run_agent(question)
    pack = _pack(seen["messages"])
    assert [item["chunk_id"] for item in pack["meeting"]] == [1, 2, 3, 4, 5, 6]
    assert pack["meeting"][1]["source_label"] == 'Meeting "Team Sync Call" on 2026-11-13, Summary'
    assert all(f"Synthetic line {item['chunk_id']}" in item["text"] for item in pack["meeting"])
    assert all("content_sha256" not in item for item in pack["meeting"])
    # The meeting's own chunks are not repeated under retrieved; other sources stay.
    assert [item["chunk_id"] for item in pack["retrieved"]] == [50, 60]
    assert pack["trello_board"]["cards"][0]["title"] == "Task alpha"
    assert "person_cards" not in pack and "person_filter" not in pack
    assert seen["loaded"] == ["fireflies/aaa.md"]
    assert seen["retrievals"] == [None]  # no meeting-scoped retrieval when the whole meeting fits
    assert "tool_loop" not in seen and len(seen["calls"]) == 1  # one structured call, no tool loop


@pytest.mark.unit
def test_retrieved_section_is_capped_at_twenty(monkeypatch: pytest.MonkeyPatch) -> None:
    question = "todos from the last sync call"
    normal = [
        SearchChunk(100 + i, f"kb/{i}.md", 0, "x", {"source_type": "kb_page"}, 0.4, content_sha256="c" * 64)
        for i in range(30)
    ]
    seen = _install(monkeypatch, question, normal)
    agent.run_agent(question)
    assert len(_pack(seen["messages"])["retrieved"]) == 20


@pytest.mark.unit
def test_budget_mode_keeps_overview_and_summary_and_stays_under_the_cap(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import functools

    from benchmark_cli import context_pack

    question = "todos from the november 13 sync call"
    big = [
        SearchChunk(
            index + 1,
            "fireflies/aaa.md",
            index,
            " ".join(f"word{index}x{n}" for n in range(60)),
            {"title": "Team Sync Call", "date": "2026-11-13", "section": "Overview" if index == 0 else "Summary" if index == 1 else "Transcript"},
            0.0,
            content_sha256="a" * 64,
        )
        for index in range(40)
    ]
    seen = _install(monkeypatch, question, [])
    monkeypatch.setattr(agent, "meeting_chunks", lambda source: list(big))
    monkeypatch.setattr(agent, "PACK_TOKEN_LIMIT", 3000)
    monkeypatch.setattr(
        agent, "budgeted_meeting_chunks", functools.partial(context_pack.budgeted_meeting_chunks, transcript_tokens=900)
    )
    with caplog.at_level("INFO"):
        agent.run_agent(question)
    pack = _pack(seen["messages"])
    ids = [item["chunk_id"] for item in pack["meeting"]]
    assert ids[:2] == [1, 2]
    assert 2 < len(ids) < 40 and ids == sorted(ids)
    assert "meeting_mode=budgeted" in caplog.text
    assert seen["retrievals"] == [None, "fireflies/aaa.md"]
    transcript_tokens = sum(context_pack.token_count(big[i - 1].chunk_text) for i in ids[2:])
    assert transcript_tokens <= 900


@pytest.mark.unit
def test_small_meeting_logs_full_mode(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    question = "todos from the november 13 sync call"
    _install(monkeypatch, question, [])
    with caplog.at_level("INFO"):
        agent.run_agent(question)
    assert "meeting_mode=full" in caplog.text


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
def test_meeting_question_without_a_resolved_meeting_adds_no_meeting_section(monkeypatch: pytest.MonkeyPatch) -> None:
    question = "who owns automating meeting notes to discord?"
    hit = _chunk(5, "fireflies/aaa.md", "Summary")
    seen = _install(monkeypatch, question, [hit])
    agent.run_agent(question)
    pack = _pack(seen["messages"])
    assert pack["meeting"] == [] and seen["loaded"] == []
    assert pack["retrieved"][0]["source_label"].startswith('Meeting "Team Sync Call"')


@pytest.mark.unit
@pytest.mark.parametrize("question", ["what did we decide about Fireflies?", "where is Discord message sync"])
def test_non_meeting_questions_add_no_meeting_section(monkeypatch: pytest.MonkeyPatch, question: str) -> None:
    hit = _chunk(5, "fireflies/aaa.md", "Summary")
    seen = _install(monkeypatch, question, [hit])
    agent.run_agent(question)
    assert seen["meeting_lists"] == 0 and seen["loaded"] == []
    assert seen["retrievals"] == [None]
    assert _pack(seen["messages"])["meeting"] == []
    assert [m["role"] for m in seen["messages"]] == ["system", "system", "user"]


@pytest.mark.unit
def test_system_prompt_covers_ownership_questions_without_inventing_owners() -> None:
    prompt = agent.SYSTEM_PROMPT.lower()
    for phrase in ("who-owns", "listed under a person", "volunteered", "assigned", "proposed", "never invent an owner"):
        assert phrase in prompt


# --------------------------------------------------------------------------- r4: abstention fixes

SUMMARY = (
    "Overview:\nTeam talked about plans.\n\nAction items:\n"
    "**Alice Example**\nAutomate structured meeting minutes (01:47:38)\nReview the pilot setup (00:12:00)\n"
    "**Bob Sample**\nBuild the larger demo project (00:30:00)\n"
)


def _sum_chunk(source: str = "fireflies/aaa.md", chunk_id: int = 7, text: str = SUMMARY) -> SearchChunk:
    return SearchChunk(chunk_id, source, 0, text, {"source_type": "fireflies"}, 0.4, content_sha256="d" * 64)


def _resp(quote: str, answer: str = "Alice Example owns it.", chunk_id: int = 7, source: str = "fireflies/aaa.md"):
    from benchmark_cli.models import AnswerResponse

    cite = Citation(source_file=source, chunk_id=chunk_id, content_sha256="d" * 64, quote=quote)
    return AnswerResponse(status="answered", answer=answer, confidence=0.9, citations=[cite])


def _quotes(response: Any) -> list[str]:
    return [c.quote for c in response.citations]


@pytest.mark.unit
@pytest.mark.parametrize("joiner", ["\n", " - ", "...", "…"])
def test_stitched_quote_splits_into_verbatim_citations(joiner: str) -> None:
    chunk = _sum_chunk()
    stitched = f"**Alice Example**{joiner}Review the pilot setup (00:12:00)"  # one line is skipped between them
    out = meeting_answers.repair_meeting_citations(_resp(stitched), {7: chunk})
    assert _quotes(out) == ["**Alice Example**", "Review the pilot setup (00:12:00)"]
    assert all(q in chunk.chunk_text for q in _quotes(out))


@pytest.mark.unit
def test_stitched_quote_drops_absent_fragments_and_none_when_all_dropped() -> None:
    chunk = _sum_chunk()
    mixed = "Something never said at all\nReview the pilot setup (00:12:00)"
    assert _quotes(meeting_answers.repair_meeting_citations(_resp(mixed), {7: chunk})) == [
        "Review the pilot setup (00:12:00)"
    ]
    assert meeting_answers.repair_meeting_citations(_resp("nothing here at all\nstill nothing here"), {7: chunk}) is None


@pytest.mark.unit
def test_repair_leaves_verbatim_non_meeting_and_no_citation_responses_alone() -> None:
    chunk = _sum_chunk()
    verbatim = _resp("Review the pilot setup (00:12:00)")
    assert _quotes(meeting_answers.repair_meeting_citations(verbatim, {7: chunk})) == _quotes(verbatim)
    kb = SearchChunk(8, "kb/page.md", 0, "line one here\nline two here", {}, 0.4, content_sha256="d" * 64)
    other = _resp("line one here\nnot there", chunk_id=8, source="kb/page.md")
    assert _quotes(meeting_answers.repair_meeting_citations(other, {8: kb})) == ["line one here\nnot there"]


@pytest.mark.unit
def test_owner_guard() -> None:
    chunk = _sum_chunk()
    ev = {7: chunk}
    item = "Automate structured meeting minutes (01:47:38)"
    assert meeting_answers.meeting_owner_conflict(_resp(item, "Bob Sample will automate minutes."), ev)
    assert not meeting_answers.meeting_owner_conflict(_resp(item, "Alice Example will automate minutes."), ev)
    assert not meeting_answers.meeting_owner_conflict(_resp(item, "Alice will automate minutes."), ev)
    plain = _sum_chunk(text="Overview:\nNo list here. Automate structured meeting minutes (01:47:38)\n")
    assert not meeting_answers.meeting_owner_conflict(_resp(item, "Bob Sample did it."), {7: plain})


@pytest.mark.unit
def test_name_check_ignores_group_phrases_only_when_a_meeting_is_loaded() -> None:
    chunk = _sum_chunk()
    ev = {7: chunk}
    answer = "The work was assigned to team members."
    assert not agent._rejected_names(answer, ev, True)
    assert agent._rejected_names(answer, ev, False)
    assert agent._rejected_names("It was assigned to Zed Qux.", ev, True)
    assert agent._rejected_names("It was assigned to Zed Qux.", ev, False)


def _run_stubbed(monkeypatch: pytest.MonkeyPatch, question: str, drafts: list[Any]) -> tuple[Any, dict[str, Any]]:
    seen = _install(monkeypatch, question, [], drafts)
    summary = _sum_chunk("fireflies/aaa.md", 1)
    monkeypatch.setattr(agent, "meeting_chunks", lambda source: [summary])
    return agent.run_agent(question), seen


@pytest.mark.unit
def test_run_agent_accepts_a_stitched_draft_with_one_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    stitched = _resp(
        "**Alice Example**\nReview the pilot setup (00:12:00)", "Alice Example owns the pilot review.", chunk_id=1
    )
    response, seen = _run_stubbed(monkeypatch, "who owns meeting minutes on november 13 sync call", [stitched])
    assert seen["outcome"] == "answered"
    assert seen["synthesis_attempts"] == 1
    assert _quotes(response) == ["**Alice Example**", "Review the pilot setup (00:12:00)"]


@pytest.mark.unit
def test_retry_carries_the_reason_for_every_question(monkeypatch: pytest.MonkeyPatch) -> None:
    wrong = _resp("Automate structured meeting minutes (01:47:38)", "Bob Sample owns minutes.", chunk_id=1)
    good = _resp("Automate structured meeting minutes (01:47:38)", "Alice Example owns minutes.", chunk_id=1)
    response, seen = _run_stubbed(monkeypatch, "who owns meeting minutes on november 13 sync call", [wrong, good])
    assert seen["outcome"] == "answered_after_citation_retry"
    assert response.answer == "Alice Example owns minutes."
    retry = seen["calls"][1][-1]["content"]
    assert retry.startswith(meeting_answers.MEETING_RETRY_PROMPT)
    assert "not listed under" in retry
    assert "single line" in meeting_answers.MEETING_RETRY_PROMPT


# --------------------------------------------------------------------------- r5: lane vs Trello

from benchmark_cli.meeting_answers import (  # noqa: E402
    asks_about_meeting_in_context,
    excludes_trello,
    resolve_meeting_in_context,
)
from benchmark_cli.trello_answers import board_question  # noqa: E402

PARENT = "what should Alex own from yesterdays meeting?"


@pytest.mark.unit
@pytest.mark.parametrize(
    "question",
    [
        "what should Alex own from yesterdays meeting?",
        "based on the Weekly Sync Test transcript, what is Alex responsible for?",
        "don't look at tickets, what did Alex agree to in the meeting",
    ],
)
def test_r5_phrasings_enter_the_lane(question: str) -> None:
    assert asks_about_meeting(question)


@pytest.mark.unit
def test_r5_yesterday_resolves_alone_and_from_the_parent() -> None:
    today = date(2026, 11, 14)
    assert parse_requested_date("what should Alex own from yesterdays meeting", today) == date(2026, 11, 13)
    resolution = resolve_meeting_in_context("and what about Sam?", PARENT, MEETINGS, today)
    assert resolution.meeting == SYNC


@pytest.mark.unit
def test_r5_follow_up_date_wins_over_the_parent() -> None:
    resolution = resolve_meeting_in_context(
        "and on november 10?", PARENT, MEETINGS, date(2026, 11, 14)
    )
    assert resolution.meeting == PLANNING
    missing = resolve_meeting_in_context("and on november 2?", PARENT, MEETINGS, date(2026, 11, 14))
    assert missing.meeting is None and missing.requested == date(2026, 11, 2)


@pytest.mark.unit
def test_r5_follow_up_lane_needs_a_meeting_parent() -> None:
    assert asks_about_meeting_in_context("and what about Sam?", PARENT)
    assert not asks_about_meeting_in_context("and what about Sam?", "what tickets does Alex have")
    assert not asks_about_meeting_in_context("and what about Sam?", None)
    assert asks_about_meeting_in_context("summarize the meeting", None)


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "where is Discord message sync",
        "what did we decide about Fireflies?",
        "sync",
        "call",
        "fireflies",
        "fathom",
    ],
)
def test_r5_negatives_stay_negative_in_context(text: str) -> None:
    assert not asks_about_meeting(text)
    assert not asks_about_meeting_in_context(text, "what tickets does Alex have")


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    ["don't look at tickets", "based only on the transcript", "only on yesterdays meeting", "ignore the trello board"],
)
def test_r5_excludes_trello_true(text: str) -> None:
    assert excludes_trello(text)
    assert excludes_trello("and Sam?", text)


@pytest.mark.unit
@pytest.mark.parametrize("text", ["what tickets does Alex have", "what is on the board", "Alex's cards"])
def test_r5_excludes_trello_false(text: str) -> None:
    assert not excludes_trello(text)
    assert not excludes_trello(text, "what tickets does Sam have")


@pytest.mark.unit
def test_r5_board_question_respects_exclusion() -> None:
    assert not board_question("don't look at tickets", PARENT)
    assert board_question("what are Alex's tickets")


def _person_install(
    monkeypatch: pytest.MonkeyPatch, question: str, drafts: list[Any], *, thread: str | None = None
) -> dict[str, Any]:
    seen = _install(monkeypatch, question, [], drafts, cards=True)
    monkeypatch.setattr(agent, "understand_query", lambda *_a, **_k: QueryPlan(searches=[question], person="Alex"))
    summary = _sum_chunk("fireflies/aaa.md", 1)
    monkeypatch.setattr(agent, "meeting_chunks", lambda source: seen["loaded"].append(source) or [summary])
    return seen


def _card_cite() -> Citation:
    return Citation(source_file="trello/c1.md", chunk_id=900, content_sha256="e" * 64, quote="Task alpha")


@pytest.mark.unit
def test_r5_person_named_meeting_answer_is_returned_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    question = "what should Alex own from yesterdays meeting?"
    good = _resp("Automate structured meeting minutes (01:47:38)", "Alice Example owns minutes.", chunk_id=1)
    seen = _person_install(monkeypatch, question, [good])
    response = agent.run_agent(question)
    assert response.answer == "Alice Example owns minutes."
    assert {c.chunk_id for c in response.citations} == {1}
    pack = _pack(seen["calls"][0])
    assert [card["title"] for card in pack["trello_board"]["cards"]] == ["Task alpha"]  # board stays in the pack
    assert "person_cards" not in pack and "person_filter" not in pack


@pytest.mark.unit
def test_r5_follow_up_uses_the_parent_for_the_meeting(monkeypatch: pytest.MonkeyPatch) -> None:
    question = "and what about Sam?"
    good = _resp("Automate structured meeting minutes (01:47:38)", "Alice Example owns minutes.", chunk_id=1)
    seen = _person_install(monkeypatch, question, [good], thread=PARENT)
    monkeypatch.setattr(agent, "today_in_new_york", lambda *_a: date(2026, 11, 14))
    response = agent.run_agent(question, thread_question=PARENT)
    assert response.answer == "Alice Example owns minutes."
    assert seen["loaded"] == ["fireflies/aaa.md"]  # the parent's "yesterdays" picked the Nov 13 meeting


@pytest.mark.unit
def test_r5_exclusion_retries_once_then_drops_trello_citations_and_keeps_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = "don't look at tickets, what does Alex own from the november 13 meeting?"
    both = _resp("Automate structured meeting minutes (01:47:38)", "Alex owns minutes and Task alpha.", chunk_id=1)
    both = both.model_copy(update={"citations": [*both.citations, _card_cite()]})
    seen = _person_install(monkeypatch, question, [both, both])
    response = agent.run_agent(question)
    assert len(seen["calls"]) == 2
    assert meeting_answers.TRELLO_EXCLUDED_RETRY in seen["calls"][1][-1]["content"]
    assert response.status == "answered"
    assert [c.chunk_id for c in response.citations] == [1]
    assert _pack(seen["calls"][0])["trello_board"]["cards"]  # the board was never removed from the pack


@pytest.mark.unit
def test_r5_exclusion_with_only_trello_citations_fails_without_a_ticket_list(monkeypatch: pytest.MonkeyPatch) -> None:
    question = "don't look at tickets, what does Alex own from the november 13 meeting?"
    from benchmark_cli.models import AnswerResponse

    bad = AnswerResponse(status="answered", answer="Alex owns Task alpha.", confidence=0.9, citations=[_card_cite()])
    seen = _person_install(monkeypatch, question, [bad, bad])
    response = agent.run_agent(question)
    assert response.status == "no_relevant_information"
    assert "Task alpha" not in response.answer
    assert seen["outcome"] == "validation_failed"


@pytest.mark.unit
def test_r5_model_failure_gives_no_ticket_list(monkeypatch: pytest.MonkeyPatch) -> None:
    question = "what should Alex own from yesterdays meeting?"
    seen = _person_install(monkeypatch, question, [None, None])
    response = agent.run_agent(question)
    assert response.status == "no_relevant_information"
    assert "Task alpha" not in response.answer
    assert seen["outcome"] == "validation_failed"
    for name in ("_deterministic_fallback", "_complete", "_model_question"):
        assert not hasattr(agent, name)


@pytest.mark.unit
def test_r5_quota_gives_the_unavailable_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    question = "what should Alex own from yesterdays meeting?"
    seen = _person_install(monkeypatch, question, [])

    def boom(**_k: Any) -> Any:
        raise agent.ModelQuotaError("rate limited")

    monkeypatch.setattr(agent, "_synthesize_answer", lambda **_k: boom())
    response = agent.run_agent(question)
    assert response.answer == agent.MODEL_UNAVAILABLE_REPLY
    assert seen["outcome"] == "model_quota"


@pytest.mark.unit
def test_r5_person_ticket_question_without_meeting_returns_the_model_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    question = "what tickets does Alex have?"
    from benchmark_cli.models import AnswerResponse

    draft = AnswerResponse(status="answered", answer="Alex has Task alpha.", confidence=0.9, citations=[_card_cite()])
    seen = _person_install(monkeypatch, question, [draft])
    response = agent.run_agent(question)
    assert response.answer == "Alex has Task alpha."
    assert [c.chunk_id for c in response.citations] == [900]
    assert [card["title"] for card in _pack(seen["calls"][0])["trello_board"]["cards"]] == ["Task alpha"]
