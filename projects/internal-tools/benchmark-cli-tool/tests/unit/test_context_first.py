"""Context-first answer path (0.1.12): pack, history, meeting resolution, light checks, threads.

Synthetic data only: Alex, Sam, "Weekly Sync Test", "Task alpha".
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import discord
import pytest

from benchmark_cli import agent, contributions, discord_ask
from benchmark_cli.context_pack import (
    Turn,
    budgeted_meeting_chunks,
    clean_history,
    compact_board,
    conversation_messages,
    history_from_starter,
)
from benchmark_cli.contributions import UNKNOWN_INVITE
from benchmark_cli.meeting_answers import (
    Meeting,
    asks_about_meeting_in_turns,
    resolve_meeting_in_turns,
)
from benchmark_cli.models import AnswerResponse, Citation
from benchmark_cli.query_understanding import QueryPlan
from benchmark_cli.retrieval import HybridSearchResult, RetrievalDecision
from benchmark_cli.storage.postgres import SearchChunk
from benchmark_cli.trello_answers import cards_from_rows

TODAY = date(2026, 11, 14)
WEEKLY_13 = Meeting("fireflies/w13.md", "Weekly Sync Test", "2026-11-13", "")
WEEKLY_11 = Meeting("fireflies/w11.md", "Weekly Sync Test", "2026-11-11", "")
MEETINGS = [WEEKLY_11, WEEKLY_13]
ALPHA = ("board", "Doing", "c1", "Task alpha", "https://example.test/c1", 900, "trello/c1.md", "e" * 64, "Task alpha card\nMembers: Alex\nLabels: sev1\nDue: 2026-12-01")
BETA = ("board", "Doing", "c2", "Task beta", "https://example.test/c2", 901, "trello/c2.md", "f" * 64, "Task beta card")
DIGEST = "a" * 64


def _meeting(source: str, count: int = 5) -> list[SearchChunk]:
    sections = ["Overview", "Summary"] + ["Transcript"] * (count - 2)
    return [
        SearchChunk(
            100 + index,
            source,
            index,
            f"Synthetic line {index}",
            {"title": "Weekly Sync Test", "date": "2026-11-13", "section": section, "source_type": "transcript"},
            0.0,
            content_sha256=DIGEST,
        )
        for index, section in enumerate(sections)
    ]


def _draft(answer: str, *citations: Citation, status: str = "answered") -> AnswerResponse:
    if status != "answered":
        return AnswerResponse(status="no_relevant_information", answer=answer, confidence=0.0, citations=[])
    return AnswerResponse(status="answered", answer=answer, confidence=0.8, citations=list(citations))


def _cite(chunk_id: int, quote: str, source: str = "fireflies/w13.md") -> Citation:
    return Citation(source_file=source, chunk_id=chunk_id, content_sha256=DIGEST, quote=quote)


def _kb(chunk_id: int = 7, text: str = "Alex owns the Task alpha plan.") -> SearchChunk:
    return SearchChunk(chunk_id, "kb/page.md", 0, text, {"source_type": "kb_page"}, 0.5, content_sha256=DIGEST)


def _install(
    monkeypatch: pytest.MonkeyPatch,
    drafts: list[Any] | None = None,
    *,
    hits: list[SearchChunk] | None = None,
    cards: list[tuple[object, ...]] | None = None,
    meetings: list[Meeting] | None = None,
) -> dict[str, Any]:
    seen: dict[str, Any] = {"calls": [], "loaded": [], "retrieved_with": []}
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **kw: seen.update(kw))
    monkeypatch.setattr(agent, "classify_query", lambda _q: None)
    monkeypatch.setattr(agent, "trello_card_rows", lambda: list(cards or []))
    monkeypatch.setattr(agent, "fts_search", lambda *_a, **_k: [])
    monkeypatch.setattr(agent, "today_in_new_york", lambda *_a: TODAY)
    monkeypatch.setattr(agent, "indexed_meetings", lambda *_a: list(MEETINGS if meetings is None else meetings))
    monkeypatch.setattr(agent, "answer_code_question", lambda *_a, **_k: None)

    def load(source: str) -> list[SearchChunk]:
        seen["loaded"].append(source)
        return _meeting(source)

    monkeypatch.setattr(agent, "meeting_chunks", load)
    monkeypatch.setattr(
        agent, "understand_query", lambda question, *_a, **_k: QueryPlan(searches=[question], person=None)
    )

    def retrieve(query: str, **kwargs: Any) -> RetrievalDecision:
        seen["retrieved_with"].append(kwargs.get("source"))
        found = hits or []
        return RetrievalDecision(
            "accepted" if found else "no_relevant_information",
            tuple(HybridSearchResult(chunk, 0.1, 0.8, 1.0) for chunk in found),
            0.8 if found else None,
            2 if found else 0,
        )

    monkeypatch.setattr(agent, "retrieve_evidence", retrieve)
    monkeypatch.setattr(
        agent,
        "call_llm",
        lambda **_k: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=""))]
        ),
    )

    def synth(*, tier: str, messages: list[dict[str, Any]]) -> Any:
        seen["calls"].append(messages)
        if not drafts:
            return None
        return drafts[min(len(seen["calls"]), len(drafts)) - 1]

    monkeypatch.setattr(agent, "_synthesize_answer", synth)
    return seen


_real_synthesize = agent._synthesize_answer


def _pack(messages: list[dict[str, Any]]) -> dict[str, Any]:
    block = next(m for m in messages if m["content"].startswith("Initial evidence"))
    return json.loads(block["content"].split("\n", 1)[1])


# 1. the pack ----------------------------------------------------------------------------------


@pytest.mark.unit
def test_compact_board_is_short_and_has_no_person_fields() -> None:
    board = compact_board(cards_from_rows([ALPHA, BETA]))
    card = board["cards"][0]
    assert set(card) == {"title", "list", "members", "labels", "due", "url", "chunk_id"}
    assert card["title"] == "Task alpha" and card["chunk_id"] == 900 and card["due"] == "2026-12-01"
    assert board["counts"] == {"Doing": 2}
    blob = json.dumps(board)
    for dropped in ("quote", "content_sha256", "source_file", "excerpt", "person_filter", "person_cards"):
        assert dropped not in blob


@pytest.mark.unit
def test_resolved_meeting_loads_every_chunk_in_order_and_the_board_is_present(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install(monkeypatch, [_draft("ok", _cite(100, "Synthetic line 0"))], cards=[ALPHA, BETA], hits=[_kb()])
    agent.run_agent("what did Alex say in the Weekly Sync Test meeting on november 13?")
    pack = _pack(seen["calls"][0])
    assert [item["chunk_id"] for item in pack["meeting"]] == [100, 101, 102, 103, 104]
    assert [item["source_label"] for item in pack["meeting"]][:2] == [
        'Meeting "Weekly Sync Test" on 2026-11-13, Overview',
        'Meeting "Weekly Sync Test" on 2026-11-13, Summary',
    ]
    assert [card["title"] for card in pack["trello_board"]["cards"]] == ["Task alpha", "Task beta"]
    assert "person_filter" not in pack and "person_cards" not in pack
    assert [item["chunk_id"] for item in pack["retrieved"]] == [7]
    assert len(seen["calls"]) == 1  # one structured call, no tool loop


@pytest.mark.unit
def test_budget_mode_keeps_overview_and_summary_and_respects_the_transcript_cap() -> None:
    big = [
        SearchChunk(
            index,
            "fireflies/w13.md",
            index,
            " ".join(f"t{index}x{n}" for n in range(40)),
            {"section": "Overview" if index == 0 else "Summary" if index == 1 else "Transcript"},
            0.0,
            content_sha256=DIGEST,
        )
        for index in range(30)
    ]
    chosen = budgeted_meeting_chunks(big, [big[10]], transcript_tokens=10_000)
    assert [chunk.id for chunk in chosen][:2] == [0, 1]
    assert {9, 10, 11} <= {chunk.id for chunk in chosen}  # the hit and one neighbour either side
    assert [chunk.id for chunk in chosen] == sorted(chunk.id for chunk in chosen)
    assert len(chosen) == 5
    tiny = budgeted_meeting_chunks(big, [big[10]], transcript_tokens=1)
    assert [chunk.id for chunk in tiny] == [0, 1]


# 3. messages ----------------------------------------------------------------------------------


@pytest.mark.unit
def test_history_arrives_as_real_turns_and_the_raw_follow_up_is_last(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install(monkeypatch, [_draft("ok", _cite(7, "Alex owns the Task alpha plan."))], hits=[_kb()])
    history = [
        Turn("user", "who owns Task alpha?"),
        Turn("assistant", "Task alpha is in the ticket list."),
    ]
    agent.run_agent("no, from the plan", history=history)
    sent = seen["calls"][0]
    assert [m["role"] for m in sent] == ["system", "system", "user", "assistant", "user"]
    assert sent[2]["content"] == "who owns Task alpha?"
    assert sent[3]["content"] == "Task alpha is in the ticket list."
    assert sent[-1]["content"] == "no, from the plan"
    assert "The thread started with" not in json.dumps(sent)
    assert not hasattr(agent, "_model_question")


@pytest.mark.unit
def test_thread_question_is_a_one_turn_history_alias() -> None:
    assert history_from_starter("  who owns   Task alpha? ") == [Turn("user", "who owns Task alpha?")]
    assert history_from_starter(None) == []
    messages = conversation_messages([Turn("user", "first")], " second ")
    assert messages == [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}]


@pytest.mark.unit
def test_history_is_capped_to_six_turns_of_1500_characters() -> None:
    turns = [Turn("user" if i % 2 == 0 else "assistant", "x" * 3000) for i in range(10)]
    cleaned = clean_history(turns)
    assert len(cleaned) == 6 and all(len(turn.text) == 1500 for turn in cleaned)
    assert clean_history([Turn("system", "no"), Turn("user", "  "), Turn("user", "kept")]) == [Turn("user", "kept")]


# 4. meeting from the thread -------------------------------------------------------------------


@pytest.mark.unit
def test_follow_up_without_a_meeting_word_resolves_from_an_earlier_user_turn() -> None:
    history = [Turn("user", "what should Alex own from the Weekly Sync Test meeting on november 13?"), Turn("assistant", "A.")]
    assert asks_about_meeting_in_turns("and Sam?", history)
    assert not asks_about_meeting_in_turns("and Sam?", [Turn("assistant", "the meeting went well")])
    assert resolve_meeting_in_turns("and Sam?", history, MEETINGS, TODAY).meeting == WEEKLY_13


@pytest.mark.unit
def test_yesterdays_resolves_against_the_day_that_turn_was_sent() -> None:
    sent = datetime(2026, 11, 12, 15, 0, tzinfo=UTC)  # two days before TODAY; "yesterday" = Nov 11
    history = [Turn("user", "what did Alex own from yesterdays meeting?", sent)]
    assert resolve_meeting_in_turns("and Sam?", history, MEETINGS, TODAY).meeting == WEEKLY_11
    undated = [Turn("user", "what did Alex own from yesterdays meeting?")]
    assert resolve_meeting_in_turns("and Sam?", undated, MEETINGS, TODAY).meeting == WEEKLY_13  # no date: today - 1


@pytest.mark.unit
def test_current_message_date_beats_an_earlier_turn() -> None:
    history = [Turn("user", "todos from the meeting on november 11?")]
    assert resolve_meeting_in_turns("and the one on november 13?", history, MEETINGS, TODAY).meeting == WEEKLY_13


@pytest.mark.unit
def test_follow_up_loads_the_meeting_named_in_an_earlier_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install(monkeypatch, [_draft("ok", _cite(100, "Synthetic line 0"))])
    history = [
        Turn("user", "what should Alex own from yesterdays meeting?", datetime(2026, 11, 12, 15, 0, tzinfo=UTC)),
        Turn("assistant", "Nothing."),
    ]
    agent.run_agent("and Sam?", history=history)
    assert seen["loaded"] == ["fireflies/w11.md"]


@pytest.mark.unit
def test_a_named_date_with_no_meeting_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install(monkeypatch, [_draft("never used", _cite(100, "x"))])
    response = agent.run_agent("and Sam?", history=[Turn("user", "todos from the october 2 meeting?")])
    assert seen["outcome"] == "meeting_not_found" and seen["calls"] == []
    assert "Weekly Sync Test (2026-11-13)" in response.answer and response.citations == []


# 5-7. nothing overwrites the model ------------------------------------------------------------


@pytest.mark.unit
def test_a_meeting_draft_that_names_no_card_is_returned_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    draft = _draft("Alex volunteered for follow-ups.", _cite(101, "Synthetic line 1"))
    _install(monkeypatch, [draft], cards=[ALPHA])
    monkeypatch.setattr(agent, "understand_query", lambda *_a, **_k: QueryPlan(searches=["Alex"], person="Alex"))
    response = agent.run_agent("what should Alex own from the Weekly Sync Test meeting on november 13?")
    assert response.answer == "Alex volunteered for follow-ups."
    assert [c.chunk_id for c in response.citations] == [101]


@pytest.mark.unit
def test_a_list_answer_that_omits_a_card_is_returned_not_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    draft = _draft("Task alpha is in Doing.", _cite(900, "Task alpha", "trello/c1.md"))
    seen = _install(monkeypatch, [draft], cards=[ALPHA, BETA])
    response = agent.run_agent("what is in Doing?")
    assert response.answer == "Task alpha is in Doing."
    assert len(seen["calls"]) == 1


@pytest.mark.unit
def test_an_unknown_list_name_still_reaches_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    draft = _draft("There is no list named Zeta.", status="no_relevant_information")
    seen = _install(monkeypatch, [draft], cards=[ALPHA, BETA])
    response = agent.run_agent("what is in the Zeta list?")
    assert len(seen["calls"]) == 1
    assert response.answer == "There is no list named Zeta."
    assert seen["outcome"] == "model_abstained"


@pytest.mark.unit
def test_a_card_citation_is_rebuilt_from_its_chunk_id(monkeypatch: pytest.MonkeyPatch) -> None:
    junk = Citation(source_file="unknown", chunk_id=900, content_sha256="0" * 64, quote="Task alpha")
    _install(monkeypatch, [_draft("Task alpha is Doing.", junk)], cards=[ALPHA])
    response = agent.run_agent("what is Task alpha's status?")
    citation = response.citations[0]
    assert (citation.source_file, citation.content_sha256) == ("trello/c1.md", "e" * 64)
    assert citation.quote in ALPHA[-1]


@pytest.mark.unit
def test_two_bad_drafts_give_validation_failed_never_a_ticket_list(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install(monkeypatch, [None, None], cards=[ALPHA, BETA])
    response = agent.run_agent("what is in Doing?")
    assert response.status == "no_relevant_information"
    assert "Task alpha" not in response.answer
    assert seen["outcome"] == "validation_failed" and len(seen["calls"]) == 2


@pytest.mark.unit
def test_quota_gives_the_unavailable_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install(monkeypatch, [], cards=[ALPHA])

    def quota(**_k: Any) -> Any:
        raise agent.ModelQuotaError("slow down")

    monkeypatch.setattr(agent, "_synthesize_answer", quota)
    response = agent.run_agent("what is in Doing?")
    assert response.answer == agent.MODEL_UNAVAILABLE_REPLY and seen["outcome"] == "model_quota"


@pytest.mark.unit
def test_an_empty_pack_abstains_without_a_model_call(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install(monkeypatch, [_draft("x", _cite(1, "x"))])
    response = agent.run_agent("what is the zorp budget?")
    assert response.status == "no_relevant_information"
    assert seen["outcome"] == "retrieval_rejected" and seen["calls"] == []


# 9. safety checks kept ------------------------------------------------------------------------


@pytest.mark.unit
def test_an_unknown_chunk_id_is_dropped_and_the_valid_citation_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    draft = _draft("Alex owns the plan.", _cite(999, "Alex owns the Task alpha plan.", "kb/page.md"), _cite(7, "Alex owns the Task alpha plan.", "kb/page.md"))
    seen = _install(monkeypatch, [draft], hits=[_kb()])
    response = agent.run_agent("who owns the plan?")
    assert [c.chunk_id for c in response.citations] == [7]
    assert len(seen["calls"]) == 1


@pytest.mark.unit
def test_only_unknown_chunk_ids_retry_then_ship_uncited(monkeypatch: pytest.MonkeyPatch) -> None:
    draft = _draft("Alex owns the plan.", _cite(999, "Alex owns the Task alpha plan.", "kb/page.md"))
    seen = _install(monkeypatch, [draft, draft], hits=[_kb()])
    response = agent.run_agent("who owns the plan?")
    assert seen["outcome"] == "answered_uncited" and len(seen["calls"]) == 2
    assert response.status == "answered" and response.answer == "Alex owns the plan."
    assert response.citations == []


@pytest.mark.unit
def test_a_paraphrased_quote_on_a_known_chunk_is_repaired_to_a_verbatim_line(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "Unrelated opening line\nAlex owns the Task alpha plan for launch\nClosing line"
    draft = _draft("Alex owns the plan.", _cite(7, "Alex is the owner of the Task alpha plan", "kb/page.md"))
    seen = _install(monkeypatch, [draft], hits=[_kb(text=text)])
    response = agent.run_agent("who owns the plan?")
    assert [c.quote for c in response.citations] == ["Alex owns the Task alpha plan for launch"]
    assert seen["outcome"] == "answered" and len(seen["calls"]) == 1


@pytest.mark.unit
def test_dangling_markers_are_stripped_from_an_uncited_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    draft = _draft("Alex owns the plan [1] and Sam reviews it [2].", _cite(999, "x", "kb/page.md"))
    _install(monkeypatch, [draft, draft], hits=[_kb()])
    response = agent.run_agent("who owns the plan?")
    assert "[" not in response.answer


@pytest.mark.unit
def test_synthesis_uses_its_own_tier_and_the_tool_loop_stays_cheap(monkeypatch: pytest.MonkeyPatch) -> None:
    tiers: dict[str, list[str]] = {"structured": [], "loop": []}
    _install(monkeypatch, [], hits=[_kb()])
    monkeypatch.setattr(agent, "_synthesize_answer", _real_synthesize)

    def structured(*, tier: str, **_k: Any) -> Any:
        tiers["structured"].append(tier)
        return agent._DraftAnswer(
            status="answered", answer="Alex owns the plan.", confidence=0.8,
            citations=[agent._DraftCitation(chunk_id=7, quote="Alex owns the Task alpha plan.")],
        )

    def loop(*, tier: str, **_k: Any) -> Any:
        tiers["loop"].append(tier)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=""))])

    monkeypatch.setattr(agent, "call_llm_structured", structured)
    monkeypatch.setattr(agent, "call_llm", loop)
    agent.run_agent("who owns the plan?")
    assert tiers == {"structured": ["synthesis"], "loop": ["cheap"]}


@pytest.mark.unit
def test_a_named_person_gets_a_person_cards_section(monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmark_cli import roster

    people = (roster.Person("Alex", ("Alex",), ("Alex",)),)
    monkeypatch.setattr(roster, "load_people", lambda: people)
    seen = _install(monkeypatch, [_draft("Alex has Task alpha.", _cite(900, "Task alpha", "trello/c1.md"))], cards=[ALPHA, BETA])
    monkeypatch.setattr(agent, "understand_query", lambda *_a, **_k: QueryPlan(searches=["Alex"], person="Alex"))
    agent.run_agent("what tickets are assigned to Alex")
    pack = _pack(seen["calls"][0])
    assert pack["person_cards"]["person"] == "Alex"
    assert [card["title"] for card in pack["person_cards"]["cards"]] == ["Task alpha"]
    assert {card["title"] for card in pack["trello_board"]["cards"]} == {"Task alpha", "Task beta"}


@pytest.mark.unit
def test_a_question_without_a_person_has_no_person_cards_key(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install(monkeypatch, [_draft("Task alpha is in Doing.", _cite(900, "Task alpha", "trello/c1.md"))], cards=[ALPHA, BETA])
    agent.run_agent("what is in Doing?")
    assert "person_cards" not in _pack(seen["calls"][0])


@pytest.mark.unit
def test_a_stitched_meeting_quote_is_split_into_verbatim_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk_text = "Alex will draft the plan\nunrelated middle line here\nSam will review the draft"
    summary = SearchChunk(100, "fireflies/w13.md", 1, chunk_text, {"section": "Summary", "title": "Weekly Sync Test", "date": "2026-11-13"}, 0.0, content_sha256=DIGEST)
    seen = _install(
        monkeypatch,
        [_draft("Alex drafts, Sam reviews.", _cite(100, "Alex will draft the plan ... Sam will review the draft"))],
    )
    monkeypatch.setattr(agent, "meeting_chunks", lambda source: [summary])
    response = agent.run_agent("todos from the Weekly Sync Test meeting on november 13?")
    assert [c.quote for c in response.citations] == ["Alex will draft the plan", "Sam will review the draft"]
    assert seen["outcome"] == "answered"


@pytest.mark.unit
def test_an_invented_url_retries_then_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    draft = _draft("See https://invented.example/page for it.", _cite(7, "Alex owns the Task alpha plan.", "kb/page.md"))
    seen = _install(monkeypatch, [draft, draft], hits=[_kb()])
    response = agent.run_agent("who owns the plan?")
    assert len(seen["calls"]) == 2
    assert "invented.example" in seen["calls"][1][-1]["content"]
    assert seen["outcome"] == "validation_failed" and response.status == "no_relevant_information"


@pytest.mark.unit
def test_an_invented_assignee_retries_then_fails_and_a_good_retry_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    cite = _cite(7, "Alex owns the Task alpha plan.", "kb/page.md")
    bad = _draft("The plan was assigned to Zed Qux.", cite)
    seen = _install(monkeypatch, [bad, bad], hits=[_kb()])
    response = agent.run_agent("who owns the plan?")
    assert seen["outcome"] == "validation_failed" and response.status == "no_relevant_information"
    good = _draft("The plan was assigned to Alex.", cite)
    seen = _install(monkeypatch, [bad, good], hits=[_kb()])
    assert agent.run_agent("who owns the plan?").answer == "The plan was assigned to Alex."
    assert seen["outcome"] == "answered_after_citation_retry"


@pytest.mark.unit
def test_a_credential_question_is_refused_before_anything_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install(monkeypatch, [_draft("x", _cite(1, "x"))])
    response = agent.run_agent("what is the api key for the sync?", history=[Turn("user", "earlier")])
    assert seen["outcome"] == "credential_refused" and seen["calls"] == []
    assert response.status == "no_relevant_information"


@pytest.mark.unit
def test_the_model_draft_schema_needs_no_hash_or_file(monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmark_cli.agent import _DraftAnswer, _DraftCitation, _to_answer

    parsed = _DraftAnswer(status="answered", answer="Alex.", confidence=0.7, citations=[_DraftCitation(chunk_id=7, quote="q")])
    converted = _to_answer(parsed)
    assert converted is not None and converted.citations[0].chunk_id == 7
    uncited = _to_answer(_DraftAnswer(status="answered", answer="No cites."))
    assert uncited is not None and uncited.status == "answered" and uncited.citations == []
    assert _to_answer(_DraftAnswer(status="no_relevant_information", answer="Unknown.", confidence=0.4)).confidence == 0.0
    monkeypatch.setattr(agent, "call_llm_structured", lambda **kw: parsed if kw["response_schema"] is _DraftAnswer else None)
    assert agent._synthesize_answer(tier="cheap", messages=[{"role": "user", "content": "q"}]) is not None


# 10. saved answers ----------------------------------------------------------------------------

SAVED_Q = "How many people are on the Task alpha team?"


def _saved(chunk_id: int = 11) -> SearchChunk:
    text = f"Question: {SAVED_Q}\nAccording to Alex on 2026-11-01:\nThree people."
    return SearchChunk(
        chunk_id,
        f"user_contributed/7/{chunk_id}.md",
        0,
        text,
        {"source_type": "user_contributed", "question": SAVED_Q, "author_name": "Alex", "contributed_at": "2026-11-01", "date": "2026-11-01"},
        0.5,
        content_sha256=DIGEST,
    )


@pytest.mark.unit
def test_exact_saved_answer_returns_before_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install(monkeypatch, [], hits=[_saved()])
    response = agent.run_agent(SAVED_Q)
    assert seen["outcome"] == "user_contributed" and seen["calls"] == []
    assert response.answer.startswith("According to Alex on 2026-11-01")


@pytest.mark.unit
def test_a_paraphrase_reaches_the_model_with_the_saved_chunk_in_saved_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contributions, "question_similarity", lambda *_a: 0.7)
    seen = _install(monkeypatch, [_draft("Three people (per Alex).", _cite(11, "Three people.", "user_contributed/7/11.md"))], hits=[_saved(), _kb()])
    response = agent.run_agent("who is on the Task alpha team?")
    assert seen.get("outcome") != "user_contributed" and len(seen["calls"]) == 1
    pack = _pack(seen["calls"][0])
    assert [item["chunk_id"] for item in pack["saved_answers"]] == [11]
    assert pack["saved_answers"][0]["source_label"] == "User contribution by Alex on 2026-11-01"
    assert [item["chunk_id"] for item in pack["retrieved"]] == [7]
    assert response.answer == "Three people (per Alex)."


@pytest.mark.unit
def test_exact_saved_answer_is_skipped_when_there_is_history(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install(monkeypatch, [_draft("Three.", _cite(11, "Three people.", "user_contributed/7/11.md"))], hits=[_saved()])
    agent.run_agent(SAVED_Q, history=[Turn("user", "about Task alpha")])
    assert len(seen["calls"]) == 1 and seen["outcome"] != "user_contributed"


@pytest.mark.unit
def test_fact_lookups_fire_only_for_a_standalone_message(monkeypatch: pytest.MonkeyPatch) -> None:
    channels = [("sample-chat", 1, "discord-sample.json", "ab" * 32, "Synthetic note.")]
    seen = _install(monkeypatch, [_draft("From the model.", _cite(7, "Alex owns the Task alpha plan.", "kb/page.md"))], hits=[_kb()])
    monkeypatch.setattr(agent, "synced_discord_channel_rows", lambda: channels)
    agent.run_agent("which channels does the bot read?")
    assert seen["outcome"] == "discord_channels"
    agent.run_agent("which channels does the bot read?", history=[Turn("user", "hi")])
    assert seen["outcome"] == "answered" and len(seen["calls"]) == 1


# 11. discord_ask ------------------------------------------------------------------------------

BOT_ID = 100000000000000010
HUMAN_ID = 100000000000000011


def _msg(message_id: int, text: str, author_id: int = HUMAN_ID, *, bot: bool = False) -> Any:
    return SimpleNamespace(
        id=message_id,
        content=text,
        author=SimpleNamespace(id=author_id, bot=bot),
        created_at=datetime(2026, 11, 13, 12, message_id % 60, tzinfo=UTC),
    )


def _thread(starter: Any | None, earlier_newest_first: list[Any]) -> Any:
    channel = MagicMock(spec=discord.Thread)
    channel.id = 5
    channel.starter_message = starter
    channel.parent = None

    async def history(**_kwargs: Any) -> Any:
        for item in earlier_newest_first:
            yield item

    channel.history = history
    return channel


@pytest.mark.unit
def test_read_thread_history_maps_authors_merges_parts_and_skips_service_replies() -> None:
    starter = _msg(1, "who owns Task alpha?")
    earlier = [  # newest first, as Discord returns it
        _msg(7, discord_ask.THANKS_REPLY, BOT_ID, bot=True),
        _msg(6, "ignored other bot", 999, bot=True),
        _msg(5, "and Sam?"),
        _msg(4, f"Alex owns it.\n\n{UNKNOWN_INVITE}", BOT_ID, bot=True),
        _msg(3, "Sources: kb/page.md", BOT_ID, bot=True),
        _msg(2, discord_ask.FAILURE_REPLY, BOT_ID, bot=True),
    ]
    message = SimpleNamespace(id=9, content="no, from the plan", channel=_thread(starter, earlier))
    turns = asyncio.run(discord_ask.read_thread_history(message, BOT_ID))
    assert [(t.role, t.text) for t in turns] == [
        ("user", "who owns Task alpha?"),
        ("assistant", "Sources: kb/page.md\nAlex owns it."),
        ("user", "and Sam?"),
    ]
    assert all(turn.sent_at is not None for turn in turns)


@pytest.mark.unit
def test_build_history_caps_turns_and_characters() -> None:
    items = [("user" if i % 2 == 0 else "assistant", f"{i}" + "x" * 3000, None) for i in range(10)]
    turns = discord_ask.build_history(items)
    assert len(turns) == 6 and turns[0].text.startswith("4") and all(len(t.text) == 1500 for t in turns)
    assert discord_ask.build_history([("user", "  ", None), ("assistant", UNKNOWN_INVITE, None)]) == []


@pytest.mark.unit
def test_read_thread_history_outside_a_thread_is_empty() -> None:
    message = SimpleNamespace(id=9, content="hi", channel=SimpleNamespace(id=1))
    assert asyncio.run(discord_ask.read_thread_history(message, BOT_ID)) == []


@pytest.mark.unit
def test_client_passes_history_to_agents_that_take_it_and_a_starter_to_older_ones() -> None:
    seen: dict[str, Any] = {}

    def with_history(question: str, history: list[Turn] | None = None) -> AnswerResponse:
        seen["history"] = history
        return _draft("Unknown.", status="no_relevant_information")

    def older(question: str, thread_question: str | None = None) -> AnswerResponse:
        seen["thread"] = thread_question
        return _draft("Unknown.", status="no_relevant_information")

    turns = [Turn("user", "first"), Turn("assistant", "reply")]
    config = discord_ask.AskConfig(100000000000000001, None)
    asyncio.run(discord_ask.AskClient(config, agent=with_history).answer_bundle("next", history=turns))
    assert seen["history"] == turns
    asyncio.run(discord_ask.AskClient(config, agent=older).answer_bundle("next", history=turns))
    assert seen["thread"] == "first"


@pytest.mark.unit
def test_ask_regress_history_turns_use_explicit_history_else_the_starter() -> None:
    from benchmark_cli.ask_regress import history_turns

    assert history_turns("and Sam?", "who owns Task alpha?") == [Turn("user", "who owns Task alpha?")]
    assert history_turns("who owns Task alpha?", "who owns Task alpha?") == []
    explicit = (("user", "q"), ("assistant", "a"))
    assert history_turns("next", None, explicit) == [Turn("user", "q"), Turn("assistant", "a")]
