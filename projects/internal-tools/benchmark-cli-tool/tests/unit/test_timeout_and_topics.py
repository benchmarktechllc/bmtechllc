"""Timeouts, topic cards, and a RAG definition that is not an ingest job."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmark_cli import agent, retrieval
from benchmark_cli.discord_ask import format_discord_messages
from benchmark_cli.grounding import named_products
from benchmark_cli.models import AnswerResponse, Citation
from benchmark_cli.retrieval import (
    HybridSearchResult,
    RetrievalDecision,
    retrieve_evidence,
    search_docs,
)
from benchmark_cli.storage.postgres import SearchChunk, query_was_canceled
from tests.unit.test_live_question_regressions import _as_tuple, _row

_DIGEST = "ab" * 32


class QueryCanceled(Exception):
    """Stand-in for psycopg.errors.QueryCanceled."""


def _note(chunk_id: int, source_type: str, text: str) -> SearchChunk:
    return SearchChunk(
        chunk_id,
        f"{source_type}/{chunk_id}.md",
        0,
        text,
        {"source_type": source_type, "title": "Sample note"},
        1.0,
        content_sha256=_DIGEST,
    )


def _quiet(**_kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=""))])


def _abstain(**_kwargs: object) -> AnswerResponse:
    return AnswerResponse(
        status="no_relevant_information",
        answer="No relevant information was found in the company document library.",
        confidence=0.0,
    )


def _install(monkeypatch: pytest.MonkeyPatch, rows: list[object]) -> None:
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "trello_card_rows", lambda: rows)
    monkeypatch.setattr(agent, "call_llm", _quiet)
    monkeypatch.setattr(agent, "call_llm_structured", _abstain)


@pytest.mark.unit
def test_query_canceled_keeps_the_other_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    note = _note(1, "kb_page", "The knowledge base portal is https://example.test/portal.")

    def vector(*_args: object, **_kwargs: object) -> list[SearchChunk]:
        raise QueryCanceled("canceling statement due to statement timeout")

    monkeypatch.setattr(retrieval, "embed_texts", lambda _texts: [[0.1, 0.2]])
    monkeypatch.setattr(retrieval, "vector_search", vector)
    monkeypatch.setattr(retrieval, "fts_search", lambda *_args, **_kwargs: [note])

    assert query_was_canceled(QueryCanceled("canceling statement due to statement timeout"))
    found = search_docs("What is the link to the knowledge base portal?")
    assert found
    assert "example.test/portal" in found[0].chunk.chunk_text

    calls: list[str] = []
    defined = _note(2, "kb_page", "benchmark-cli-tool is the retrieval-only RAG CLI.")

    def search(query: str, *_args: object, **_kwargs: object) -> list[HybridSearchResult]:
        calls.append(query)
        if len(calls) == 1:
            raise QueryCanceled("canceling statement due to statement timeout")
        return [HybridSearchResult(defined, 0.4, vector_score=0.4, fts_score=1.0)]

    monkeypatch.setattr(retrieval, "search_docs", search)
    decision = retrieve_evidence("what's the RAG tool?")
    assert len(calls) > 1
    assert decision.status == "accepted"
    assert "retrieval-only RAG CLI" in decision.results[0].chunk.chunk_text


@pytest.mark.unit
def test_portal_link_uses_one_search(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def search(query: str, *_args: object, **kwargs: object) -> list[HybridSearchResult]:
        calls.append(query)
        assert kwargs.get("source_type") is None
        return []

    monkeypatch.setattr(retrieval, "search_docs", search)
    retrieve_evidence("What is the link to the knowledge base portal?")
    assert calls == ["What is the link to the knowledge base portal?"]


@pytest.mark.unit
def test_retrieval_stops_at_the_time_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def search(query: str, *_args: object, **_kwargs: object) -> list[HybridSearchResult]:
        calls.append(query)
        return []

    monkeypatch.setattr(retrieval, "search_docs", search)
    retrieve_evidence("what's the RAG tool?", budget_seconds=0)
    assert calls == ["what's the RAG tool?"]


@pytest.mark.unit
def test_topic_words_are_not_named_products() -> None:
    assert named_products("is there anything about security on our plate?") == ()
    assert named_products("what's left for the kiosk?") == ()
    assert named_products("what's the deal with the used car project?") == ()
    assert named_products("what model should we use for high reasoning work?") == ()
    assert named_products("what did we decide about Fireflies?") == ("fireflies",)


@pytest.mark.unit
def test_security_kiosk_and_used_car_use_matching_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    security = _row("Doing", "Sample security review", "sec", 1)
    kiosk = _row("Doing", "Sample kiosk pilot", "kiosk", 2)
    done = _row("Done", "Sample kiosk archive", "old", 3)
    used = _row("Backlog", "Sample used car project", "car", 4)
    note = _note(9, "transcript", "The kiosk pilot still needs a sample lobby install.")
    _install(
        monkeypatch,
        [_as_tuple(security), _as_tuple(kiosk), _as_tuple(done), _as_tuple(used)],
    )
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision(
            "accepted",
            (HybridSearchResult(note, 0.2, vector_score=0.2, fts_score=1.0),),
            0.2,
            2,
        ),
    )

    plate = agent.run_agent("is there anything about security on our plate?")
    assert plate.status == "answered"
    assert "Sample security review" in plate.answer
    assert "Sample kiosk pilot" not in plate.answer

    left = agent.run_agent("what's left for the kiosk?")
    assert "Sample kiosk pilot" in left.answer
    assert "lobby install" in left.answer
    assert "Sample kiosk archive" not in left.answer

    deal = agent.run_agent("what's the deal with the used car project?")
    assert "Sample used car project" in deal.answer


@pytest.mark.unit
def test_high_reasoning_model_quotes_the_note(monkeypatch: pytest.MonkeyPatch) -> None:
    note = _note(8, "discord", "Use Opus when the work needs high reasoning.")
    _install(monkeypatch, [])
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision(
            "accepted",
            (HybridSearchResult(note, 0.2, vector_score=0.2, fts_score=1.0),),
            0.2,
            3,
        ),
    )

    response = agent.run_agent("what model should we use for high reasoning work?")

    assert response.status == "answered"
    assert "Opus" in response.answer
    assert response.citations[0].quote == note.chunk_text


@pytest.mark.unit
def test_modest_cosine_keeps_a_topic_note(monkeypatch: pytest.MonkeyPatch) -> None:
    note = _note(8, "discord", "Use Opus when the work needs high reasoning.")

    def search(query: str, *_args: object, **_kwargs: object) -> list[HybridSearchResult]:
        if "benchmark-cli-tool" in query.casefold():
            return []
        return [HybridSearchResult(note, 0.2, vector_score=0.2, fts_score=1.0)]

    monkeypatch.setattr(retrieval, "search_docs", search)
    decision = retrieve_evidence(
        "what model should we use for high reasoning work?",
        min_cosine_similarity=0.30,
    )
    assert decision.status == "accepted"
    assert "Opus" in decision.results[0].chunk.chunk_text


@pytest.mark.unit
def test_rag_tool_prefers_a_description_over_ingest_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    jobs = _note(1, "trello", "Ingest jobs include discord sync and benchmark-cli-tool.")
    defined = _note(
        2,
        "kb_page",
        "benchmark-cli-tool is the retrieval-only RAG CLI. It searches indexed company documents.",
    )
    _install(monkeypatch, [])
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision(
            "accepted",
            (
                HybridSearchResult(jobs, 0.5, vector_score=0.5, fts_score=1.0),
                HybridSearchResult(defined, 0.4, vector_score=0.4, fts_score=1.0),
            ),
            0.5,
            2,
        ),
    )
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: AnswerResponse(
            status="answered",
            answer="- [Ingest benchmark-cli-tool](https://trello.com/c/job)",
            confidence=0.8,
            citations=[
                Citation(
                    source_file=jobs.source_file,
                    chunk_id=jobs.id,
                    content_sha256=_DIGEST,
                    quote=jobs.chunk_text,
                )
            ],
        ),
    )

    defined_answer = agent.run_agent("what's the RAG tool?")
    assert "retrieval-only" in defined_answer.answer
    assert "searches" in defined_answer.answer
    assert "Ingest jobs" not in defined_answer.answer

    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision(
            "accepted",
            (HybridSearchResult(jobs, 0.5, vector_score=0.5, fts_score=1.0),),
            0.5,
            1,
        ),
    )
    unknown = agent.run_agent("what's the RAG tool?")
    assert unknown.status == "no_relevant_information"
    assert "do not define" in unknown.answer
    assert "Add answer" in "\n".join(format_discord_messages(unknown))


@pytest.mark.unit
def test_certificate_mentions_a_short_course(monkeypatch: pytest.MonkeyPatch) -> None:
    note = _note(
        4,
        "kb_page",
        "No certificate URL is indexed.\nAgentic AI in 38 min.",
    )
    _install(monkeypatch, [])
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision(
            "accepted",
            (HybridSearchResult(note, 0.4, vector_score=0.4, fts_score=1.0),),
            0.4,
            1,
        ),
    )
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: AnswerResponse(
            status="answered",
            answer="No certificate link is known.",
            confidence=0.4,
            citations=[
                Citation(
                    source_file=note.source_file,
                    chunk_id=note.id,
                    content_sha256=_DIGEST,
                    quote="No certificate URL is indexed.",
                )
            ],
        ),
    )

    response = agent.run_agent("what was the link to the Anthropic certificate on agentic AI")
    assert "example.test" not in response.answer
    assert "Agentic AI in 38 min" in response.answer
