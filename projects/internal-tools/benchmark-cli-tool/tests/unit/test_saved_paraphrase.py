"""Second-tier saved-answer match: question signature plus a cosine floor. Synthetic rows only."""

from __future__ import annotations

import hashlib

import pytest

from benchmark_cli import agent, contributions
from benchmark_cli.ask_regress import CODE_CASES
from benchmark_cli.contributions import (
    PARAPHRASE_MIN_SIMILARITY,
    QUESTION_MATCH_SIMILARITY,
    contribution_answer,
    contribution_match,
    same_question,
)
from benchmark_cli.query_understanding import QueryPlan
from benchmark_cli.retrieval import HybridSearchResult, RetrievalDecision
from benchmark_cli.storage.postgres import SearchChunk

SAVED = "How many employees does BMTech currently have?"

POSITIVES = (
    "how many people work at bmtech",
    "what's our headcount",
    "how many employees do we have",
    "how big is the bmtech team",
    "how many poeple work at bmtech",
    "how many employes does bmtech have",
    "how many people work at bmtec",
    "How many staff do we have?",
    "how many members does BMTech have right now",
    "how many people are on the team",
    "how many people does bm tech employ",
    "what is BMTech's headcount",
    "how many team members are there at bmtech",
    "hey, how many people work at BMTech?",
    "how large is our team",
    "number of employees at bmtech",
    "bmtech headcount",
    "how many ppl work at bmtech",
    "what is the size of the team",
    "how many people are employed at bmtech",
)

NEGATIVES = (
    "Who is on the core team?",
    "who's on the team?",
    "What projects is BMTech currently working on?",
    "What channels has BMTech synced?",
    "How many cards are in Backlog?",
    "how many tickets are done vs open?",
    "how many tickets does bmtech have",
    "how many people are in #general",
    "who works at bmtech",
    "how many clients do we have",
    "how many people work on the kiosk",
    "how many people are on the core team",
    "how many people joined bmtech",
    "how many people left bmtech",
    "how many people did we hire",
    "how many people worked at bmtech",
    "how many employees did bmtech have",
    "how many employees will we have next year",
    "how many employees does Acme have",
    "how many employees at bmtech and at acme",
    "how many people work there",
    "how many people work on this",
    "how many contractors does bmtech have",
    "how many projects does bmtech have",
    "how many customers do we have",
    "how many teams do we have",
    "how many numbers do we have",
    "how many markers do we have",
    "how many embers do we have",
    "how many employers do we have",
    "how big is bmtech",
    "how many",
    "do we have many employees",
    "how much staff budget do we have",
)


def _row(chunk_id: int, question: str, *, when: str = "2026-01-02", body: str = "Sample count is 3.") -> HybridSearchResult:
    text = f"Question: {question}\nAccording to Ada Example on {when}:\n{body}"
    chunk = SearchChunk(
        chunk_id,
        f"user_contributed/7/{chunk_id}.md",
        0,
        text,
        {"source_type": "user_contributed", "question": question, "author_name": "Ada Example", "contributed_at": when, "date": when},
        0.5,
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
    return HybridSearchResult(chunk, 0.2, vector_score=0.5, fts_score=1.0)


def _similarity(monkeypatch: pytest.MonkeyPatch, value: float) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    def fake(asked: str, saved: str) -> float:
        calls.append((asked, saved))
        return value

    monkeypatch.setattr(contributions, "question_similarity", fake)
    return calls


# --------------------------------------------------------------------------- signature (T3)


@pytest.mark.unit
@pytest.mark.parametrize("question", POSITIVES)
def test_headcount_paraphrases_share_the_signature(question: str) -> None:
    assert same_question(question, SAVED)


@pytest.mark.unit
@pytest.mark.parametrize("question", NEGATIVES)
def test_other_questions_do_not_share_the_signature(question: str) -> None:
    assert not same_question(question, SAVED)


@pytest.mark.unit
def test_word_order_matters_for_remaining_terms() -> None:
    assert not same_question("who reports to sam", "who does sam report to")
    assert same_question("who reports to sam", "Who reports to Sam?")


# --------------------------------------------------------------------------- matcher (T4, T6)


@pytest.mark.unit
@pytest.mark.parametrize("question", POSITIVES)
def test_paraphrase_tier_matches_below_the_exact_threshold(monkeypatch: pytest.MonkeyPatch, question: str) -> None:
    calls = _similarity(monkeypatch, QUESTION_MATCH_SIMILARITY - 0.01)
    match = contribution_match(question, (_row(11, SAVED),), allow_paraphrase=True)
    assert match is not None and match.tier == "paraphrase" and match.chunk_id == 11
    assert match.response.answer == "According to Ada Example on 2026-01-02: Sample count is 3."
    assert len(match.response.citations) == 1
    assert len(calls) == 1
    # Contract of the exact-only entry point is unchanged.
    assert contribution_answer(question, (_row(11, SAVED),)) is None
    assert contribution_match(question, (_row(11, SAVED),)) is None


@pytest.mark.unit
@pytest.mark.parametrize("question", NEGATIVES + tuple(case["q"] for case in CODE_CASES))
def test_paraphrase_tier_rejects_other_questions_at_worst_case_similarity(
    monkeypatch: pytest.MonkeyPatch, question: str
) -> None:
    _similarity(monkeypatch, QUESTION_MATCH_SIMILARITY - 0.01)
    assert contribution_match(question, (_row(11, SAVED),), allow_paraphrase=True) is None


@pytest.mark.unit
def test_paraphrase_tier_needs_the_cosine_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    _similarity(monkeypatch, PARAPHRASE_MIN_SIMILARITY - 0.01)
    assert contribution_match(POSITIVES[0], (_row(11, SAVED),), allow_paraphrase=True) is None
    _similarity(monkeypatch, PARAPHRASE_MIN_SIMILARITY)
    assert contribution_match(POSITIVES[0], (_row(11, SAVED),), allow_paraphrase=True) is not None


@pytest.mark.unit
def test_exact_tier_wins_and_newest_paraphrase_row_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(asked: str, saved: str) -> float:
        return 0.9 if saved == "how many people work here" else 0.6

    monkeypatch.setattr(contributions, "question_similarity", fake)
    rows = (
        _row(11, SAVED, when="2026-03-01"),
        _row(12, "how many people work here", when="2026-01-01", body="Other."),
    )
    match = contribution_match("how many people work at bmtech", rows, allow_paraphrase=True)
    assert match is not None and match.tier == "exact" and match.chunk_id == 12

    _similarity(monkeypatch, 0.6)
    rows = (_row(11, SAVED, when="2026-01-01"), _row(12, "what's our headcount", when="2026-02-01", body="Four."))
    match = contribution_match("how many staff do we have", rows, allow_paraphrase=True)
    assert match is not None and match.tier == "paraphrase" and match.chunk_id == 12


@pytest.mark.unit
def test_normalised_equal_question_makes_no_similarity_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _similarity(monkeypatch, 0.0)
    match = contribution_match("how many employees does bmtech currently have", (_row(11, SAVED),), allow_paraphrase=True)
    assert match is not None and match.tier == "exact" and match.similarity is None
    assert calls == []


# --------------------------------------------------------------------------- agent call site (T5)


def _agent(
    monkeypatch: pytest.MonkeyPatch,
    question: str,
    *,
    person: str | None = None,
    thread_question: str | None = None,
    rows: tuple[HybridSearchResult, ...] | None = None,
) -> tuple[object, dict[str, object]]:
    seen: dict[str, object] = {}
    _similarity(monkeypatch, 0.7)
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **kwargs: seen.update(kwargs))
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [])
    monkeypatch.setattr(agent, "understand_query", lambda *_a, **_k: QueryPlan(searches=[question], person=person))
    monkeypatch.setattr(agent, "mentioned_person", lambda *_a, **_k: person)
    hits = rows if rows is not None else (_row(11, SAVED),)
    monkeypatch.setattr(agent, "retrieve_evidence", lambda *_a, **_k: RetrievalDecision("accepted", hits, 0.7, 2))

    def model_down(**_kwargs: object) -> object:
        seen["model_called"] = True
        raise RuntimeError("model down")

    monkeypatch.setattr(agent, "call_llm", model_down)
    monkeypatch.setattr(agent, "call_llm_structured", model_down)
    monkeypatch.setattr(agent, "answer_code_question", lambda *_a, **_k: None)
    try:
        response = agent.run_agent(question, thread_question=thread_question)
    except RuntimeError:
        response = None
    return response, seen


@pytest.mark.unit
@pytest.mark.parametrize("question", ("how many people work at bmtech", "what's our headcount"))
def test_agent_answers_a_paraphrase_from_the_saved_row_without_the_model(
    monkeypatch: pytest.MonkeyPatch, question: str
) -> None:
    response, seen = _agent(monkeypatch, question)
    assert seen["outcome"] == "user_contributed_paraphrase" and "model_called" not in seen
    assert response.answer.startswith("According to Ada Example on 2026-01-02: ")
    assert response.citations[0].source_file == "user_contributed/7/11.md"


@pytest.mark.unit
def test_agent_exact_wording_stays_on_the_exact_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    _, seen = _agent(monkeypatch, SAVED)
    assert seen["outcome"] == "user_contributed"


@pytest.mark.unit
def test_agent_vetoes_keep_their_own_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    # Same-signature saved row for "core team" exists; the veto keeps the core-team shortcut.
    core = (_row(11, "who's on the core team"),)
    _, seen = _agent(monkeypatch, "Who is on the core team?", rows=core)
    assert seen["outcome"] == "core_team"
    # A named person, a distinct thread starter, and a code question never take tier 2.
    _, seen = _agent(monkeypatch, "how many people work at bmtech", person="Sample Person")
    assert seen.get("outcome") != "user_contributed_paraphrase"
    _, seen = _agent(monkeypatch, "how many people", thread_question="Who posts in the sample channel?")
    assert seen.get("outcome") != "user_contributed_paraphrase"
    code_question = CODE_CASES[0]["q"]
    code_row = (_row(11, code_question.rstrip("?") + " exactly?"),)
    _, seen = _agent(monkeypatch, code_question, rows=code_row)
    assert seen.get("outcome") != "user_contributed_paraphrase"


def _other_row(chunk_id: int) -> HybridSearchResult:
    text = "Sample channel chatter with no saved answer."
    chunk = SearchChunk(
        chunk_id, "discord-channel-7.json", 0, text, {"source_type": "discord"}, 0.5,
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )  # fmt: skip
    return HybridSearchResult(chunk, 0.2, vector_score=0.5, fts_score=1.0)


def _fts(monkeypatch: pytest.MonkeyPatch, rows: tuple[HybridSearchResult, ...]) -> list[str]:
    calls: list[str] = []

    def fake(phrase: str, *, limit: int) -> list[SearchChunk]:
        calls.append(phrase)
        return [row.chunk for row in rows]

    monkeypatch.setattr(agent, "fts_search", fake)
    return calls


@pytest.mark.unit
def test_headcount_question_finds_the_saved_row_through_the_employees_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Hybrid retrieval misses the saved row; the headcount shortcut's "employees" lookup has it.
    calls = _fts(monkeypatch, (_row(11, SAVED),))
    response, seen = _agent(monkeypatch, "what's our headcount", rows=(_other_row(5),))
    assert seen["outcome"] == "user_contributed_paraphrase" and "model_called" not in seen
    assert response.citations[0].source_file == "user_contributed/7/11.md"
    assert calls == ["employees"]


@pytest.mark.unit
def test_headcount_lookup_runs_once_and_only_when_hybrid_has_no_match(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fts(monkeypatch, (_other_row(6),))
    _, seen = _agent(monkeypatch, "what's our headcount", rows=(_other_row(5),))
    assert seen["outcome"].startswith("employee_count_") and calls == ["employees"]
    calls = _fts(monkeypatch, (_row(12, SAVED),))
    _, seen = _agent(monkeypatch, "what's our headcount")
    assert seen["outcome"] == "user_contributed_paraphrase" and calls == []
    # Not a headcount question: no extra lookup, no saved answer from it.
    calls = _fts(monkeypatch, (_row(12, SAVED),))
    _, seen = _agent(monkeypatch, "how many people are on the team", rows=(_other_row(5),))
    assert seen.get("outcome") != "user_contributed_paraphrase" and calls == []
