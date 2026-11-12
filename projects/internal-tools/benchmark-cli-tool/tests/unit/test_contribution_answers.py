"""Saved answers are used before the headcount and core-team shortcuts."""

from __future__ import annotations

import hashlib

import pytest

from benchmark_cli import agent
from benchmark_cli.models import AnswerResponse
from benchmark_cli.retrieval import HybridSearchResult, RetrievalDecision
from benchmark_cli.storage.postgres import SearchChunk

_DIGEST = hashlib.sha256(b"synthetic-contribution").hexdigest()
_TEAM_DIGEST = hashlib.sha256(b"synthetic-team").hexdigest()
_WIFI_DIGEST = hashlib.sha256(b"synthetic-wifi").hexdigest()


def _chunk(
    chunk_id: int,
    source: str,
    text: str,
    metadata: dict[str, object],
    *,
    digest: str,
    vector: float = 0.2,
) -> SearchChunk:
    return SearchChunk(chunk_id, source, 0, text, metadata, vector, content_sha256=digest)


def _contribution(
    chunk_id: int,
    source: str,
    question: str,
    body: str,
    *,
    author: str = "Ada Example",
    when: str = "2026-11-12",
    digest: str = _DIGEST,
    vector: float = 0.84,
) -> SearchChunk:
    text = f"Question: {question}\nAccording to {author} on {when}:\n{body}"
    return _chunk(
        chunk_id,
        source,
        text,
        {
            "source_type": "user_contributed",
            "question": question,
            "author_name": author,
            "contributed_at": when,
            "date": when,
        },
        digest=digest,
        vector=vector,
    )


def _trello() -> SearchChunk:
    return _chunk(
        3,
        "trello/tIvVZqio/sample.md",
        "Card: Sample portal chore\nList: Doing\nURL: https://trello.com/c/sample",
        {"source_type": "trello", "title": "Sample portal chore"},
        digest=hashlib.sha256(b"trello-noise").hexdigest(),
        vector=0.4,
    )


def _run(monkeypatch: pytest.MonkeyPatch, question: str, hits: tuple[SearchChunk, ...]) -> AnswerResponse:
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [])
    monkeypatch.setattr(agent, "call_llm", lambda **_kwargs: pytest.fail("contribution answers must not call the model"))
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: pytest.fail("contribution answers must not call the model"),
    )
    results = tuple(
        HybridSearchResult(chunk, 0.2, vector_score=chunk.score, fts_score=1.0) for chunk in hits
    )
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision("accepted", results, 0.84, 2),
    )
    return agent.run_agent(question)


@pytest.mark.unit
def test_headcount_uses_a_saved_answer_before_the_stock_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    saved = _contribution(
        8802,
        "user_contributed/7/9.md",
        "How many employees does BMTech currently have?",
        "Sample count is 3.\nSample note stays visible.",
    )

    response = _run(monkeypatch, "How many employees does BMTech currently have ?", (saved, _trello()))

    assert response.status == "answered"
    assert "Sample count is 3." in response.answer
    assert "Sample note stays visible." in response.answer
    assert "According to" in response.answer
    assert response.citations[0].source_file == "user_contributed/7/9.md"


@pytest.mark.unit
def test_core_team_uses_a_saved_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    saved = _contribution(
        1400,
        "user_contributed/7/10.md",
        "Who is on the core team?",
        "The core team is Ada Example and Ben Stone.",
        digest=_TEAM_DIGEST,
        when="2026-11-11",
    )

    response = _run(monkeypatch, "who is on the core team", (saved, _trello()))

    assert response.status == "answered"
    assert "According to" in response.answer
    assert "Ada Example" in response.answer
    assert response.citations[0].source_file.startswith("user_contributed/")
    assert "does not name a core team" not in response.answer.casefold()


@pytest.mark.unit
def test_headcount_paraphrase_matches_on_question_similarity(monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmark_cli.contributions import (
        QUESTION_MATCH_SIMILARITY,
        normalize_question,
        question_similarity,
    )

    saved_question = "How many employees does BMTech currently have?"
    asked = "how many people work at bmtech"
    other = "Tickets assigned to Dante"
    fanual = "Tickets assigned to Fanual"
    table = {
        normalize_question(saved_question): [1.0, 0.0],
        normalize_question(asked): [0.90, 0.40],
        normalize_question(other): [0.0, 1.0],
        normalize_question(fanual): [0.55, 0.55],
    }

    def embed(texts: list[str]) -> list[list[float]]:
        return [table[normalize_question(text)] for text in texts]

    monkeypatch.setattr("benchmark_cli.contributions.embed_texts", embed)
    assert question_similarity(asked, saved_question) >= QUESTION_MATCH_SIMILARITY
    assert question_similarity(fanual, other) < QUESTION_MATCH_SIMILARITY

    saved = _contribution(
        8803,
        "user_contributed/7/11.md",
        saved_question,
        "We have 3 people",
        digest=hashlib.sha256(b"paraphrase").hexdigest(),
        vector=0.2,
    )
    response = _run(monkeypatch, asked, (saved, _trello()))

    assert response.status == "answered"
    assert "3" in response.answer
    assert "According to" in response.answer
    assert "We have 3 people" in response.answer


@pytest.mark.unit
def test_non_shortcut_question_reuses_a_saved_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    saved = _contribution(
        1500,
        "user_contributed/7/12.md",
        "what's our office wifi name?",
        "The office wifi name is SampleNet.",
        digest=_WIFI_DIGEST,
    )

    response = _run(monkeypatch, "what's our office wifi name?", (saved, _trello()))

    assert response.status == "answered"
    assert "SampleNet" in response.answer
    assert "According to" in response.answer
    assert response.citations[0].source_file == "user_contributed/7/12.md"


def _noise_contributions() -> tuple[SearchChunk, ...]:
    """High chunk scores, unrelated saved questions. These must not become the answer."""
    dante = _contribution(
        8801,
        "user_contributed/1/1.md",
        "Tickets assigned to Dante",
        "Dante is dantino12",
        author="Ada Example",
        digest=hashlib.sha256(b"dante-note").hexdigest(),
        vector=0.93,
    )
    headcount = _contribution(
        8802,
        "user_contributed/1/2.md",
        "How many employees does BMTech currently have?",
        "Sample count is 3.",
        author="Ada Example",
        digest=hashlib.sha256(b"count-note").hexdigest(),
        vector=0.91,
        when="2026-11-11",
    )
    return (dante, headcount, _trello())


def _far_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Different questions embed far apart, so only a real paraphrase can match."""
    from benchmark_cli.contributions import normalize_question

    seen: dict[str, list[float]] = {}

    def embed(texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            key = normalize_question(text)
            if key not in seen:
                vector = [0.0] * 32
                vector[len(seen)] = 1.0
                seen[key] = vector
            vectors.append(seen[key])
        return vectors

    monkeypatch.setattr("benchmark_cli.contributions.embed_texts", embed)


@pytest.mark.unit
@pytest.mark.parametrize(
    "question",
    (
        "Tickets assigned to Fanual",
        "tickets for fano",
        "what's dantino working on",
        "What tickets are we currently working on?",
        "Which tickets are still open?",
    ),
)
def test_unrelated_contributions_do_not_override_board_answers(
    monkeypatch: pytest.MonkeyPatch, question: str
) -> None:
    _far_questions(monkeypatch)
    from benchmark_cli.query_understanding import QueryPlan
    from benchmark_cli.replay import sample_board

    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "trello_card_rows", lambda: list(sample_board()))
    monkeypatch.setattr(
        agent,
        "call_llm",
        lambda **_kwargs: __import__("types").SimpleNamespace(
            choices=[__import__("types").SimpleNamespace(message=__import__("types").SimpleNamespace(tool_calls=None, content=""))]
        ),
    )
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: AnswerResponse(
            status="no_relevant_information",
            answer="No relevant information was found.",
            confidence=0.0,
            citations=[],
        ),
    )
    person = None
    folded = question.casefold()
    if "fanual" in folded or "fano" in folded:
        person = "Fanuel Mengistu"
    elif "dantino" in folded:
        person = "Dante Paniccia"
    monkeypatch.setattr(
        agent,
        "understand_query",
        lambda *_args, **_kwargs: QueryPlan(searches=[question], person=person),
    )
    hits = _noise_contributions()
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision(
            "accepted",
            tuple(HybridSearchResult(chunk, 0.2, vector_score=chunk.score, fts_score=1.0) for chunk in hits),
            0.93,
            2,
        ),
    )

    response = agent.run_agent(question)

    assert response.status == "answered"
    assert "According to" not in response.answer
    assert all(not citation.source_file.startswith("user_contributed/") for citation in response.citations)
    if person == "Fanuel Mengistu":
        assert "Kiosk MVP" in response.answer
    elif person == "Dante Paniccia":
        assert "Ship the portal" in response.answer
    else:
        assert "Ship the portal" in response.answer


@pytest.mark.unit
def test_headcount_without_a_matching_contribution_stays_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _far_questions(monkeypatch)
    response = _run(monkeypatch, "How many employees does BMTech currently have?", _noise_contributions()[:1])

    assert response.status == "no_relevant_information"
    assert "does not state a current employee count" in response.answer.casefold()
    assert "According to" not in response.answer

