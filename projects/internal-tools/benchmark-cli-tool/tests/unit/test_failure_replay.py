"""Offline reproduction of the af24ae7 live failures.

Synthetic cards use raw Trello member names. The model is stubbed. ``run_agent``
still runs validation, citation repair, and the snapshot fallback.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from benchmark_cli import agent
from benchmark_cli.models import AnswerResponse, Citation
from benchmark_cli.query_understanding import QueryPlan
from benchmark_cli.replay import sample_board
from benchmark_cli.retrieval import HybridSearchResult, RetrievalDecision
from benchmark_cli.storage.postgres import SearchChunk

_WATERMARK = "The image watermark lives at https://example.com/watermark-sample"
_DIGEST = hashlib.sha256(b"watermark-chunk").hexdigest()


def _quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "classify_query", lambda _question: "cheap")
    monkeypatch.setattr(
        agent,
        "call_llm",
        lambda **_kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=""))]
        ),
    )
    monkeypatch.setattr(agent, "trello_card_rows", lambda: list(sample_board()))
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision("no_relevant_information", (), 0.12, 0),
    )


def _plan(person: str | None, *searches: str):
    def _understand(_question: str, _thread: str | None = None) -> QueryPlan:
        return QueryPlan(searches=list(searches) or ["sample"], person=person)

    return _understand


def _draft(answer: str, citations: list[Citation] | None = None, *, status: str = "answered") -> AnswerResponse:
    if status != "answered":
        return AnswerResponse(status="no_relevant_information", answer=answer, confidence=0.0, citations=[])
    return AnswerResponse(status="answered", answer=answer, confidence=0.8, citations=citations or [])


@pytest.mark.unit
def test_backlog_abstention_still_answers_from_the_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    _quiet(monkeypatch)
    monkeypatch.setattr(agent, "understand_query", _plan(None, "What's in Backlog?"))
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: _draft("There is no relevant information provided to answer what is in the Backlog.", status="unknown"),
    )

    response = agent.run_agent("What's in Backlog?")

    assert response.status == "answered"
    assert "Review the lobby copy" in response.answer
    assert "Ship the portal" not in response.answer


@pytest.mark.unit
def test_board_summary_abstention_still_uses_the_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    _quiet(monkeypatch)
    monkeypatch.setattr(agent, "understand_query", _plan(None, "summarize the board"))
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: _draft(
            "I cannot summarize the board as there is insufficient evidence to provide a detailed overview.",
            status="unknown",
        ),
    )

    response = agent.run_agent("summarize the board for me")

    assert response.status == "answered"
    assert "Ship the portal" in response.answer
    assert "Review the lobby copy" in response.answer


@pytest.mark.unit
def test_dante_draft_with_a_bad_citation_and_partial_count_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _quiet(monkeypatch)
    monkeypatch.setattr(agent, "understand_query", _plan("Dante Paniccia", "Dante Paniccia tickets"))
    ship = sample_board()[0]
    answer = (
        "Dante Paniccia is assigned to "
        "[Ship the portal](https://trello.com/c/ship) on Doing and "
        "[Old portal import](https://trello.com/c/old) on Done. Doing 2."
    )
    bad = Citation(
        source_file=ship.source_file,
        chunk_id=ship.chunk_id,
        content_sha256=ship.content_sha256,
        quote="This quote is not in the card.",
    )
    monkeypatch.setattr(agent, "call_llm_structured", lambda **_kwargs: _draft(answer, [bad]))

    response = agent.run_agent("What tickets are assigned to Dante")

    assert response.status == "answered"
    assert "Dante Paniccia" in response.answer
    assert "Ship the portal" in response.answer
    assert "Old portal import" in response.answer
    assert "Kiosk MVP" not in response.answer
    assert response.citations


@pytest.mark.unit
def test_fanual_surface_name_does_not_drop_fanuels_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    _quiet(monkeypatch)
    monkeypatch.setattr(agent, "understand_query", _plan("Fanuel Mengistu", "Fanuel Mengistu tickets"))
    kiosk = sample_board()[2]
    answer = (
        "Tickets assigned to Fanual: "
        "[Kiosk MVP](https://trello.com/c/kiosk) and "
        "[Retired kiosk notes](https://trello.com/c/retired)."
    )
    bad = Citation(
        source_file=kiosk.source_file,
        chunk_id=kiosk.chunk_id,
        content_sha256=kiosk.content_sha256,
        quote="This quote is not in the card.",
    )
    monkeypatch.setattr(agent, "call_llm_structured", lambda **_kwargs: _draft(answer, [bad]))

    response = agent.run_agent("Tickets assigned to Fanual")

    assert response.status == "answered"
    assert "Kiosk MVP" in response.answer
    assert "Retired kiosk notes" in response.answer
    assert "Ship the portal" not in response.answer


@pytest.mark.unit
def test_anwar_follow_up_does_not_keep_dantes_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    _quiet(monkeypatch)
    monkeypatch.setattr(
        agent,
        "understand_query",
        _plan("Anwar Hussein", "Anwar Hussein tickets"),
    )
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: _draft(
            "No relevant information was found in the company document library.",
            status="unknown",
        ),
    )

    response = agent.run_agent(
        "Tickets assigned to Anwar",
        thread_question="What tickets are assigned to Dante",
    )

    assert response.status == "answered"
    assert "Review the lobby copy" in response.answer
    assert "Ship the portal" not in response.answer


@pytest.mark.unit
def test_partial_working_on_draft_keeps_done_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    _quiet(monkeypatch)
    monkeypatch.setattr(agent, "understand_query", _plan("Dante Paniccia", "Dante Paniccia"))
    ship = sample_board()[0]
    answer = "Dante Paniccia is working on [Ship the portal](https://trello.com/c/ship)."
    citation = Citation(
        source_file=ship.source_file,
        chunk_id=ship.chunk_id,
        content_sha256=ship.content_sha256,
        quote="Card: Ship the portal",
    )
    monkeypatch.setattr(agent, "call_llm_structured", lambda **_kwargs: _draft(answer, [citation]))

    response = agent.run_agent("what's dantino working on")

    assert response.status == "answered"
    assert "Ship the portal" in response.answer
    assert "Old portal import" in response.answer
    assert "Kiosk MVP" not in response.answer


@pytest.mark.unit
def test_unrelated_link_is_not_returned_as_the_requested_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _quiet(monkeypatch)
    chunk = SearchChunk(
        90,
        "discord-sample.json",
        0,
        _WATERMARK,
        {"source_type": "discord", "channel": "sample"},
        0.4,
        content_sha256=_DIGEST,
    )
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision(
            "accepted",
            (HybridSearchResult(chunk, 0.2, vector_score=0.4, fts_score=1.0),),
            0.4,
            1,
        ),
    )
    monkeypatch.setattr(agent, "understand_query", _plan(None, "certificate link"))
    answer = f"The certificate link is {_WATERMARK.split()[-1]}"
    citation = Citation(
        source_file=chunk.source_file,
        chunk_id=chunk.id,
        content_sha256=_DIGEST,
        quote=_WATERMARK,
    )
    monkeypatch.setattr(agent, "call_llm_structured", lambda **_kwargs: _draft(answer, [citation]))

    response = agent.run_agent("what was the link to the certificate")

    assert response.status == "no_relevant_information"
    assert "example.com" not in response.answer


@pytest.mark.unit
def test_every_chat_tier_uses_the_cheap_model() -> None:
    from benchmark_cli.config import EMBEDDING_MODEL, get_model_config
    from benchmark_cli.llm import _chat_tier
    from benchmark_cli.router import classify_query

    assert _chat_tier("flagship") == "cheap"
    assert _chat_tier("cheap") == "cheap"
    assert _chat_tier("embedding") == "embedding"
    assert get_model_config(_chat_tier("flagship")).model == get_model_config("cheap").model
    assert get_model_config("embedding").model == EMBEDDING_MODEL
    assert classify_query("Explain the hard multi-step case") == "cheap"
