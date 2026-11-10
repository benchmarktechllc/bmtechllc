"""Certificate links, synonym hits, channel wording, and open-versus-done counts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmark_cli import agent
from benchmark_cli.grounding import asks_for_synced_channels
from benchmark_cli.models import AnswerResponse, Citation
from benchmark_cli.retrieval import (
    HybridSearchResult,
    RetrievalDecision,
    chunk_matches_synonym,
    retrieve_evidence,
)
from benchmark_cli.storage.postgres import SearchChunk
from tests.unit.test_live_question_regressions import _as_tuple, _row

_DIGEST = "cd" * 32


def _noop_llm(**_kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=""))])


def _chunk(chunk_id: int, source_type: str, text: str) -> SearchChunk:
    return SearchChunk(
        chunk_id,
        f"{source_type}/{chunk_id}.md",
        0,
        text,
        {"source_type": source_type, "title": "Sample note"},
        1.0,
        content_sha256=_DIGEST,
    )


@pytest.mark.unit
def test_certificate_url_must_describe_the_certificate(monkeypatch: pytest.MonkeyPatch) -> None:
    news = (
        "Anthropic published a claude text watermark article at "
        "https://example.test/claude-text-watermark."
    )
    course = "A related sample course covers agentic AI."
    note = _chunk(3, "kb_page", f"{news}\n{course}")
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [])
    monkeypatch.setattr(agent, "call_llm", _noop_llm)
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision(
            "accepted",
            (HybridSearchResult(note, 0.4, vector_score=0.4, fts_score=1.0),),
            0.4,
            2,
        ),
    )
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: AnswerResponse(
            status="answered",
            answer="The certificate is at https://example.test/claude-text-watermark.",
            confidence=0.9,
            citations=[
                Citation(
                    source_file=note.source_file,
                    chunk_id=note.id,
                    content_sha256=_DIGEST,
                    quote=news,
                )
            ],
        ),
    )

    response = agent.run_agent("what was the link to the Anthropic/Claude certificate on agentic AI")

    assert "claude-text-watermark" not in response.answer
    assert "No certificate link is known." in response.answer
    assert "sample course" in response.answer
    assert response.citations
    assert response.citations[0].quote == course


@pytest.mark.unit
def test_rag_tool_and_fireflies_synonyms_reach_discord_and_kb(monkeypatch: pytest.MonkeyPatch) -> None:
    rag = _chunk(1, "kb_page", "benchmark-cli-tool is the retrieval-only RAG CLI.")
    notes = _chunk(2, "discord", "The team decided to adopt the meeting notetaker.")

    def search(query: str, *_args: object, **_kwargs: object) -> list[HybridSearchResult]:
        folded = query.casefold()
        if "benchmark-cli-tool" in folded or "rag cli" in folded:
            return [HybridSearchResult(rag, 0.05, vector_score=0.05, fts_score=1.0)]
        if "notetaker" in folded or "fireflies" in folded:
            return [HybridSearchResult(notes, 0.05, vector_score=0.05, fts_score=1.0)]
        return []

    monkeypatch.setattr("benchmark_cli.retrieval.search_docs", search)
    tool = retrieve_evidence("what's the RAG tool?", min_cosine_similarity=0.30)
    meeting = retrieve_evidence("what did we decide about Fireflies?", min_cosine_similarity=0.30)

    assert tool.status == "accepted"
    assert chunk_matches_synonym("what's the RAG tool?", tool.results[0].chunk.chunk_text)
    assert "benchmark-cli-tool" in tool.results[0].chunk.chunk_text
    assert meeting.status == "no_relevant_information"
    assert not chunk_matches_synonym(
        "what did we decide about Fireflies?",
        "The team decided to adopt the meeting notetaker.",
    )
    assert chunk_matches_synonym(
        "what did we decide about Fireflies?",
        "The team decided to keep Fireflies.",
    )


@pytest.mark.unit
def test_bot_reads_channels_and_sev_meaning_are_both_answered(monkeypatch: pytest.MonkeyPatch) -> None:
    assert asks_for_synced_channels("which channels does the bot read")
    sev = _row(
        "Sev 2",
        "Sample sev two card",
        "s2",
        2,
        chunk_text=(
            "Card: Sample sev two card\nList: Sev 2\n"
            "Sev 2 means a medium severity item.\n"
            "URL: https://trello.com/c/s2"
        ),
    )
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [_as_tuple(sev)])
    monkeypatch.setattr(
        agent,
        "synced_discord_channel_rows",
        lambda: [("sample-portal", 1, "discord-sample.json", "ab" * 32, "Synthetic portal note.")],
    )
    monkeypatch.setattr(agent, "call_llm", _noop_llm)
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision("no_relevant_information", (), None, 0),
    )
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: AnswerResponse(
            status="answered",
            answer="- [Sample sev two card](https://trello.com/c/s2)",
            confidence=0.8,
            citations=[
                Citation(
                    source_file=sev.source_file,
                    chunk_id=sev.chunk_id,
                    content_sha256=sev.content_sha256,
                    quote="Card: Sample sev two card",
                )
            ],
        ),
    )

    response = agent.run_agent("which channels does the bot read and what's in Sev 2?")

    assert "#sample-portal" in response.answer
    assert "Sev 2" in response.answer

    meaning = agent.run_agent("what does Sev 2 mean?")
    assert "severity" in meaning.answer.casefold() or "more urgent" in meaning.answer.casefold()
    assert "medium severity" in meaning.answer
    assert "Sample sev two card" in meaning.answer


@pytest.mark.unit
def test_done_versus_open_counts_every_non_done_list(monkeypatch: pytest.MonkeyPatch) -> None:
    done = [_row("Done", f"Sample done {index}", f"z{index}", index + 1) for index in range(2)]
    doing = _row("Doing", "Sample doing card", "d1", 10)
    backlog = _row("Backlog", "Sample backlog card", "b1", 11)
    sev = _row("Sev 1", "Sample sev card", "s1", 12)
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(
        agent,
        "trello_card_rows",
        lambda: [_as_tuple(card) for card in (*done, doing, backlog, sev)],
    )
    monkeypatch.setattr(agent, "call_llm", _noop_llm)
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision("no_relevant_information", (), None, 0),
    )
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: AnswerResponse(
            status="answered",
            answer="Done has 2 cards.\n- [Sample done 0](https://trello.com/c/z0)",
            confidence=0.8,
            citations=[
                Citation(
                    source_file=done[0].source_file,
                    chunk_id=done[0].chunk_id,
                    content_sha256=done[0].content_sha256,
                    quote="Card: Sample done 0",
                )
            ],
        ),
    )

    response = agent.run_agent("how many tickets are done vs open?")

    assert "Done has 2 cards." in response.answer
    assert "Open has 3 cards" in response.answer
    assert "Doing 1" in response.answer
    assert "Backlog 1" in response.answer
    assert "Sev 1 1" in response.answer
    assert "Sample doing card" in response.answer
