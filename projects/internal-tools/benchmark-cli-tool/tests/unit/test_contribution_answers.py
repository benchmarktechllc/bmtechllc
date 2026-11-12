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
        1346,
        "user_contributed/7/9.md",
        "How many employees does BMTech currently have?",
        "There are 3 members of the team",
    )

    response = _run(monkeypatch, "How many employees does BMTech currently have ?", (saved, _trello()))

    assert response.status == "answered"
    assert "3" in response.answer
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
def test_headcount_paraphrase_uses_the_top_contribution(monkeypatch: pytest.MonkeyPatch) -> None:
    saved = _contribution(
        1347,
        "user_contributed/7/11.md",
        "How many people does BMTech have?",
        "We have 3 people",
        digest=hashlib.sha256(b"paraphrase").hexdigest(),
        vector=0.81,
    )

    response = _run(monkeypatch, "How many employees does BMTech currently have?", (saved, _trello()))

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
