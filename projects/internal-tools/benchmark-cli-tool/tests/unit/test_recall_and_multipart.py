"""Paraphrase recall, multi-part answers, and complete board summaries."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmark_cli import agent
from benchmark_cli.contributions import UNKNOWN_INVITE
from benchmark_cli.models import AnswerResponse, Citation
from benchmark_cli.retrieval import (
    HybridSearchResult,
    RetrievalDecision,
    expand_queries,
    retrieve_evidence,
    split_subquestions,
)
from benchmark_cli.storage.postgres import SearchChunk

_DIGEST = "ab" * 32


def _hit(chunk_id: int, text: str, source_type: str, *, cosine: float) -> HybridSearchResult:
    chunk = SearchChunk(
        chunk_id,
        f"{source_type}/{chunk_id}.md",
        0,
        text,
        {"source_type": source_type, "title": "Sample page"},
        1.0,
        content_sha256=_DIGEST,
    )
    return HybridSearchResult(chunk, 0.2, vector_score=cosine, fts_score=1.0)


def _noop_llm(**_kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=""))])


@pytest.mark.unit
def test_synonym_expansion_finds_the_rag_tool_and_kb_portal(monkeypatch: pytest.MonkeyPatch) -> None:
    assert any("benchmark-cli-tool" in query for query in expand_queries("what's the RAG tool?"))
    assert any("Discord bot" in query for query in expand_queries("how does the KB portal work?"))
    assert any("meeting transcript" in query.casefold() for query in expand_queries("what did we decide about Fireflies?"))

    def search(query: str, *_args: object, **_kwargs: object) -> list[HybridSearchResult]:
        folded = query.casefold()
        if "benchmark-cli-tool" in folded or "retrieval-only" in folded:
            return [
                _hit(
                    1,
                    "benchmark-cli-tool is the retrieval-only RAG CLI.",
                    "kb_page",
                    cosine=0.2,
                )
            ]
        if "discord bot" in folded or "knowledge base portal" in folded:
            return [
                _hit(
                    2,
                    "The sample portal serves the knowledge base. The Discord bot reads it.",
                    "kb_page",
                    cosine=0.2,
                )
            ]
        if "fireflies" in folded:
            return [
                _hit(
                    3,
                    "The sample meeting decided to keep Fireflies for transcripts.",
                    "transcript",
                    cosine=0.18,
                )
            ]
        return []

    monkeypatch.setattr("benchmark_cli.retrieval.search_docs", search)

    rag = retrieve_evidence("what's the RAG tool?", min_cosine_similarity=0.30, min_keyword_overlap=1)
    portal = retrieve_evidence("how does the KB portal work?", min_cosine_similarity=0.30, min_keyword_overlap=1)
    meeting = retrieve_evidence(
        "what did we decide about Fireflies?",
        min_cosine_similarity=0.30,
        min_keyword_overlap=1,
    )

    assert rag.status == "accepted"
    assert "retrieval-only RAG CLI" in rag.results[0].chunk.chunk_text
    assert portal.status == "accepted"
    assert "Discord bot" in portal.results[0].chunk.chunk_text
    assert meeting.status == "accepted"
    assert "Fireflies" in meeting.results[0].chunk.chunk_text


@pytest.mark.unit
def test_multipart_answers_channels_and_names_the_unknown_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert split_subquestions("Which channels are synced and what does Sev 2 mean?") == [
        "Which channels are synced",
        "what does Sev 2 mean",
    ]
    from tests.unit.test_live_question_regressions import _DIGEST as live_digest
    from tests.unit.test_live_question_regressions import _as_tuple, _row

    sev = _row("Sev 2", "Sample severity note", "s2", 4)
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [_as_tuple(sev)])
    monkeypatch.setattr(
        agent,
        "synced_discord_channel_rows",
        lambda: [("sample-portal", 1, "discord-sample.json", live_digest, "Synthetic portal note.")],
    )
    monkeypatch.setattr(agent, "call_llm", _noop_llm)
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision("no_relevant_information", (), None, 0),
    )

    def only_severity(**_kwargs: object) -> AnswerResponse:
        return AnswerResponse(
            status="answered",
            answer="Sev 2 is the sample severity list.",
            confidence=0.8,
            citations=[
                Citation(
                    source_file=sev.source_file,
                    chunk_id=sev.chunk_id,
                    content_sha256=sev.content_sha256,
                    quote="Card: Sample severity note",
                )
            ],
        )

    monkeypatch.setattr(agent, "call_llm_structured", only_severity)
    both = agent.run_agent("Which channels are synced and what does Sev 2 mean?")
    assert "#sample-portal" in both.answer
    assert "Sev 2" in both.answer

    def only_channels(**_kwargs: object) -> AnswerResponse:
        return AnswerResponse(
            status="answered",
            answer="Synced Discord channels include #sample-portal.",
            confidence=0.8,
            citations=[
                Citation(
                    source_file=sev.source_file,
                    chunk_id=sev.chunk_id,
                    content_sha256=sev.content_sha256,
                    quote="Card: Sample severity note",
                )
            ],
        )

    monkeypatch.setattr(agent, "call_llm_structured", only_channels)
    partial = agent.run_agent("Which channels are synced and what is the sample cafeteria policy?")
    assert "#sample-portal" in partial.answer
    assert "I don't know this part" in partial.answer
    assert "cafeteria" in partial.answer.casefold()
    assert UNKNOWN_INVITE in partial.answer


@pytest.mark.unit
def test_board_overview_includes_sev_lists_when_the_model_skips_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.unit.test_live_question_regressions import _as_tuple, _row

    doing = _row("Doing", "Sample doing card", "d1", 1)
    sev = _row("Sev 2", "Sample sev card", "s2", 2)
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [_as_tuple(doing), _as_tuple(sev)])
    monkeypatch.setattr(agent, "call_llm", _noop_llm)
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision("no_relevant_information", (), None, 0),
    )

    def incomplete(**_kwargs: object) -> AnswerResponse:
        return AnswerResponse(
            status="answered",
            answer="Doing 1\n- [Sample doing card](https://trello.com/c/d1)",
            confidence=0.9,
            citations=[
                Citation(
                    source_file=doing.source_file,
                    chunk_id=doing.chunk_id,
                    content_sha256=doing.content_sha256,
                    quote="Card: Sample doing card",
                )
            ],
        )

    monkeypatch.setattr(agent, "call_llm_structured", incomplete)
    response = agent.run_agent("What's on the Trello board?")

    assert "Sev 2 1" in response.answer
    assert "Doing 1" in response.answer
    assert "Sample sev card" in response.answer
