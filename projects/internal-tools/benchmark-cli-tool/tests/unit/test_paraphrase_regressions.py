"""Wording variants that must still answer from the snapshot or a mapped chunk."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmark_cli import agent
from benchmark_cli.discord_ask import format_discord_messages
from benchmark_cli.models import AnswerResponse, Citation
from benchmark_cli.retrieval import HybridSearchResult, RetrievalDecision, retrieve_evidence
from benchmark_cli.storage.postgres import SearchChunk
from tests.unit.test_live_question_regressions import _as_tuple, _row

_DIGEST = "ef" * 32

_BOARD_WORDINGS = (
    "summarize the board for me",
    "what's on the board?",
    "give me a summary of the board",
    "overview of the trello board",
    "what is on the board right now",
)
_DOING_WORDINGS = (
    "which trello tickets are in Doing",
    "what's in Doing?",
    "show the Doing list",
    "what cards are in Doing",
    "tickets in the Doing column",
)
_RAG_WORDINGS = (
    "what's the RAG tool?",
    "what is the rag tool",
    "which tool is the RAG tool",
    "tell me about the RAG tool",
    "what do we use as the RAG tool",
)
_SEV_WORDINGS = (
    "what does Sev 2 mean?",
    "what is Sev 2",
    "explain Sev 2",
    "what does severity 2 mean",
    "Sev 2 meaning",
)


def _abstain(**_kwargs: object) -> AnswerResponse:
    return AnswerResponse(
        status="no_relevant_information",
        answer="No relevant information was found in the company document library.",
        confidence=0.0,
    )


def _quiet_completion(**_kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=""))])


def _install(monkeypatch: pytest.MonkeyPatch, rows: list[object]) -> None:
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "trello_card_rows", lambda: rows)
    monkeypatch.setattr(agent, "call_llm", _quiet_completion)
    monkeypatch.setattr(agent, "call_llm_structured", _abstain)
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision("no_relevant_information", (), None, 0),
    )


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


@pytest.mark.unit
@pytest.mark.parametrize("question", _BOARD_WORDINGS)
def test_board_summary_paraphrases_use_the_snapshot(
    monkeypatch: pytest.MonkeyPatch, question: str
) -> None:
    cards = [
        _row("Backlog", "Sample backlog card", "b1", 1),
        _row("Doing", "Sample doing card", "d1", 2),
        _row("Done", "Sample done card", "z1", 3),
        _row("Sev 1", "Sample sev one card", "s1", 4),
        _row("Sev 2", "Sample sev two card", "s2", 5),
    ]
    _install(monkeypatch, [_as_tuple(card) for card in cards])

    response = agent.run_agent(question)

    assert response.status == "answered"
    assert "company document library" not in response.answer
    for label, count in (("Backlog", 1), ("Doing", 1), ("Done", 1), ("Sev 1", 1), ("Sev 2", 1)):
        assert f"{label} {count}" in response.answer


@pytest.mark.unit
@pytest.mark.parametrize("question", _DOING_WORDINGS)
def test_doing_list_paraphrases_list_the_doing_cards(
    monkeypatch: pytest.MonkeyPatch, question: str
) -> None:
    doing = [_row("Doing", f"Sample doing {index}", f"d{index}", index + 1) for index in range(4)]
    other = _row("Backlog", "Sample backlog card", "b1", 20)
    _install(monkeypatch, [_as_tuple(card) for card in (*doing, other)])

    response = agent.run_agent(question)

    assert response.status == "answered"
    for card in doing:
        assert card.title in response.answer
    assert "Sample backlog card" not in response.answer


@pytest.mark.unit
@pytest.mark.parametrize("question", _SEV_WORDINGS)
def test_sev_meaning_paraphrases_explain_and_list_cards(
    monkeypatch: pytest.MonkeyPatch, question: str
) -> None:
    sev = _row(
        "Sev 2",
        "Sample sev two card",
        "s2",
        2,
        chunk_text="Card: Sample sev two card\nList: Sev 2\nSev 2 means a medium severity item.",
    )
    _install(monkeypatch, [_as_tuple(sev)])

    response = agent.run_agent(question)

    assert response.status == "answered"
    assert "more urgent" in response.answer.casefold()
    assert "Sample sev two card" in response.answer
    assert "medium severity" in response.answer


@pytest.mark.unit
def test_sev_membership_stays_a_card_list(monkeypatch: pytest.MonkeyPatch) -> None:
    sev = _row("Sev 2", "Sample sev two card", "s2", 2)
    _install(monkeypatch, [_as_tuple(sev)])

    response = agent.run_agent("what's in Sev 2?")

    assert "Sample sev two card" in response.answer
    assert "more urgent" not in response.answer.casefold()


@pytest.mark.unit
@pytest.mark.parametrize("question", _RAG_WORDINGS)
def test_rag_tool_paraphrases_quote_the_indexed_chunk(
    monkeypatch: pytest.MonkeyPatch, question: str
) -> None:
    note = _note(7, "kb_page", "benchmark-cli-tool is the retrieval-only RAG CLI.")
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [])
    monkeypatch.setattr(agent, "call_llm", _quiet_completion)
    monkeypatch.setattr(agent, "call_llm_structured", _abstain)
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

    response = agent.run_agent(question)

    assert response.status == "answered"
    assert "benchmark-cli-tool" in response.answer
    assert response.citations[0].quote == note.chunk_text


@pytest.mark.unit
def test_rag_cli_chunks_are_not_dropped_by_the_source_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    """The original wording fills every source slot before the mapped name is searched."""
    noise = [
        _note(index, "discord" if index % 2 else "kb_page", f"Unrelated catalog note {index}.")
        for index in range(1, 21)
    ]
    rag_chunks = [
        _note(100, "kb_page", "benchmark-cli-tool is the retrieval-only CLI."),
        _note(101, "kb_page", "One note calls it the RAG CLI."),
        _note(102, "discord", "Another note calls it the RAG CLI."),
        _note(103, "kb_page", "A third note calls it the RAG CLI."),
        _note(104, "kb_page", "A fourth note calls it the RAG CLI."),
    ]

    def search(query: str, *_args: object, **kwargs: object) -> list[HybridSearchResult]:
        folded = query.casefold()
        source_type = kwargs.get("source_type")
        if "benchmark-cli-tool" in folded or "rag cli" in folded or "cli-tool" in folded:
            return [
                HybridSearchResult(chunk, 0.4, vector_score=0.2, fts_score=1.0)
                for chunk in rag_chunks
                if source_type in (None, chunk.metadata["source_type"])
            ][:4]
        batch = [
            HybridSearchResult(chunk, 0.5, vector_score=0.55, fts_score=0.2)
            for chunk in noise
            if source_type in (None, chunk.metadata["source_type"])
        ]
        return batch[:4]

    monkeypatch.setattr("benchmark_cli.retrieval.search_docs", search)

    decision = retrieve_evidence("what's the RAG tool?", min_cosine_similarity=0.30)

    texts = [result.chunk.chunk_text for result in decision.results]
    assert decision.status == "accepted"
    assert any("benchmark-cli-tool" in text for text in texts)
    assert sum("RAG CLI" in text for text in texts) >= 1


@pytest.mark.unit
def test_fireflies_without_evidence_still_abstains(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, [])

    response = agent.run_agent("what did we decide about Fireflies?")

    assert response.status == "no_relevant_information"
    assert "notetaker" not in response.answer.casefold()


@pytest.mark.unit
def test_fireflies_synonym_cannot_ground_a_notion_transcript(monkeypatch: pytest.MonkeyPatch) -> None:
    note = _note(
        9,
        "discord",
        "We decided to pull meeting transcripts from Notion.",
    )
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [])
    monkeypatch.setattr(agent, "call_llm", _quiet_completion)
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision(
            "accepted",
            (HybridSearchResult(note, 0.8, vector_score=0.8, fts_score=1.0),),
            0.8,
            2,
        ),
    )
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: AnswerResponse(
            status="answered",
            answer="We decided to pull meeting transcripts from Notion.",
            confidence=0.8,
            citations=[
                Citation(
                    source_file=note.source_file,
                    chunk_id=note.id,
                    content_sha256=_DIGEST,
                    quote=note.chunk_text,
                )
            ],
        ),
    )

    response = agent.run_agent("what did we decide about Fireflies?")

    assert response.status == "no_relevant_information"
    assert "Notion" not in response.answer
    assert "Fireflies" not in response.answer
    rendered = "\n".join(format_discord_messages(response))
    assert "Add answer" in rendered


@pytest.mark.unit
def test_synced_channels_and_sev_meaning_follow_the_standalone_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sev = _row(
        "Sev 2",
        "Sample sev two card",
        "s2",
        2,
        chunk_text=(
            "Card: Sample sev two card\nList: Sev 2\n"
            "Sev 2 items are not as critical as those classified under"
        ),
    )
    _install(monkeypatch, [_as_tuple(sev)])
    monkeypatch.setattr(
        agent,
        "synced_discord_channel_rows",
        lambda: [("sample-portal", 1, "discord-sample.json", "ab" * 32, "Synthetic portal note.")],
    )
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: AnswerResponse(
            status="answered",
            answer=(
                "The synced channels are unknown. "
                "Sev 2 is not as critical as those classified under"
            ),
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

    response = agent.run_agent("which channels are synced and what does Sev 2 mean?")

    assert "#sample-portal" in response.answer
    assert "more urgent" in response.answer.casefold()
    assert "unknown" not in response.answer.casefold()
    assert "classified under" not in response.answer
    assert response.answer.rstrip()[-1] in ".!?" or response.answer.rstrip().endswith(")")


@pytest.mark.unit
def test_explain_sev_does_not_end_mid_sentence(monkeypatch: pytest.MonkeyPatch) -> None:
    sev = _row(
        "Sev 2",
        "Sample sev two card",
        "s2",
        2,
        chunk_text=(
            "Card: Sample sev two card\nList: Sev 2\n"
            "Sev 2 items are not as critical as those classified under"
        ),
    )
    _install(monkeypatch, [_as_tuple(sev)])
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: AnswerResponse(
            status="answered",
            answer="Sev 2 items are not as critical as those classified under",
            confidence=0.8,
            citations=[
                Citation(
                    source_file=sev.source_file,
                    chunk_id=sev.chunk_id,
                    content_sha256=sev.content_sha256,
                    quote="Sev 2 items are not as critical as those classified under",
                )
            ],
        ),
    )

    response = agent.run_agent("explain Sev 2")

    assert response.status == "answered"
    assert "more urgent" in response.answer.casefold()
    assert "classified under" not in response.answer
    assert not response.answer.rstrip().endswith("under")


@pytest.mark.unit
def test_rag_tool_answer_is_a_definition_not_an_ingest_list(monkeypatch: pytest.MonkeyPatch) -> None:
    jobs = _note(1, "kb_page", "Ingest jobs include discord sync and benchmark-cli-tool.")
    defined = _note(
        2,
        "kb_page",
        "benchmark-cli-tool is the retrieval-only RAG CLI. It searches indexed company documents.",
    )
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [])
    monkeypatch.setattr(agent, "call_llm", _quiet_completion)
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision(
            "accepted",
            (
                HybridSearchResult(jobs, 0.4, vector_score=0.4, fts_score=1.0),
                HybridSearchResult(defined, 0.3, vector_score=0.3, fts_score=1.0),
            ),
            0.4,
            2,
        ),
    )
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: AnswerResponse(
            status="answered",
            answer=jobs.chunk_text,
            confidence=0.7,
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

    response = agent.run_agent("what's the RAG tool?")

    assert response.status == "answered"
    assert "retrieval-only" in response.answer
    assert "searches" in response.answer
    assert "Ingest jobs" not in response.answer


@pytest.mark.unit
def test_portal_how_keeps_the_explanation(monkeypatch: pytest.MonkeyPatch) -> None:
    home = _note(3, "kb_page", "The knowledge base portal is https://example.test/portal.")
    guide = _note(
        4,
        "kb_page",
        "The portal uses in-browser markdown forms. "
        "Submit creates a pull request or makes the content live. "
        "Token auth protects the forms.",
    )
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [])
    monkeypatch.setattr(agent, "call_llm", _quiet_completion)
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision(
            "accepted",
            (
                HybridSearchResult(home, 0.5, vector_score=0.5, fts_score=1.0),
                HybridSearchResult(guide, 0.4, vector_score=0.4, fts_score=1.0),
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
            answer="The knowledge base portal is https://example.test/portal.",
            confidence=0.8,
            citations=[
                Citation(
                    source_file=home.source_file,
                    chunk_id=home.id,
                    content_sha256=_DIGEST,
                    quote=home.chunk_text,
                )
            ],
        ),
    )

    response = agent.run_agent("how does the KB portal work?")

    assert "markdown forms" in response.answer
    assert "pull request" in response.answer
    assert "Token auth" in response.answer
    assert response.answer.strip() != home.chunk_text


@pytest.mark.unit
def test_portal_how_chunks_survive_url_only_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    noise = [
        _note(index, "kb_page", f"The knowledge base portal is https://example.test/{index}.")
        for index in range(1, 17)
    ]
    guide = _note(
        40,
        "kb_page",
        "The portal uses in-browser markdown forms. Submit creates a pull request.",
    )

    def search(query: str, *_args: object, **kwargs: object) -> list[HybridSearchResult]:
        source_type = kwargs.get("source_type")
        folded = query.casefold()
        hits = []
        if "markdown" in folded or "form" in folded or source_type in (None, "kb_page"):
            if "markdown" in guide.chunk_text.casefold() and (
                "how" in folded or "portal" in folded or "knowledge base" in folded
            ):
                hits.append(HybridSearchResult(guide, 0.3, vector_score=0.25, fts_score=1.0))
        hits.extend(
            HybridSearchResult(chunk, 0.6, vector_score=0.6, fts_score=0.5)
            for chunk in noise
            if source_type in (None, "kb_page")
        )
        return hits[:4]

    monkeypatch.setattr("benchmark_cli.retrieval.search_docs", search)
    decision = retrieve_evidence("how does the KB portal work?", min_cosine_similarity=0.30)
    texts = " ".join(result.chunk.chunk_text for result in decision.results)
    assert "markdown forms" in texts
