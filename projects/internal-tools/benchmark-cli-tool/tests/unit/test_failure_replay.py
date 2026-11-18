"""Offline reproduction of the af24ae7 live failures.

Synthetic cards use raw Trello member names. The model is stubbed. ``run_agent``
still runs validation, citation repair, and the snapshot fallback.
"""

from __future__ import annotations

import hashlib
import json
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
    def _understand(_question: str, _thread: str | None = None, **_kwargs: object) -> QueryPlan:
        return QueryPlan(searches=list(searches) or ["sample"], person=person)

    return _understand


def _draft(answer: str, citations: list[Citation] | None = None, *, status: str = "answered") -> AnswerResponse:
    if status != "answered":
        return AnswerResponse(status="no_relevant_information", answer=answer, confidence=0.0, citations=[])
    return AnswerResponse(status="answered", answer=answer, confidence=0.8, citations=citations or [])


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
def test_unrelated_link_triggers_one_retry_hint_without_substitution(
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
    sent: list[str] = []

    def structured(**kwargs: object) -> AnswerResponse:
        sent.append(json.dumps(kwargs["messages"]))
        return _draft(answer, [citation])

    monkeypatch.setattr(agent, "call_llm_structured", structured)

    response = agent.run_agent("what was the link to the certificate")

    # 0.1.12: an unsupported document link is a retry hint, never a substitution.
    assert len(sent) == 2
    assert "does not support it" in sent[1]
    assert response.status == "answered"


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


def _sev_board():
    from benchmark_cli.trello_answers import TrelloCardRow

    rows = list(sample_board())
    card_id = "sev"
    rows.append(
        TrelloCardRow(
            board="Sample Board",
            list_name="Sev 2",
            card_id=card_id,
            title="Sample sev two note",
            url="https://trello.com/c/sev",
            chunk_id=7,
            source_file="trello/tIvVZqio/sev.md",
            content_sha256=hashlib.sha256(card_id.encode()).hexdigest(),
            chunk_text="Card: Sample sev two note\nList: Sev 2\nURL: https://trello.com/c/sev",
        )
    )
    return rows


@pytest.mark.unit
def test_raw_question_is_always_searched(monkeypatch: pytest.MonkeyPatch) -> None:
    _quiet(monkeypatch)
    note = "Aug 28 notes: the kiosk specifications remain."
    chunk = SearchChunk(
        80,
        "meeting-sample.md",
        0,
        note,
        {"source_type": "transcript", "meeting": "sample", "title": "Aug 28"},
        0.8,
        content_sha256=hashlib.sha256(b"kiosk-note").hexdigest(),
    )

    def retrieve(query: str, **kwargs: object) -> RetrievalDecision:
        queries = list(kwargs.get("queries") or [query])
        if any(item.casefold() == "what's left for the kiosk?" for item in queries):
            return RetrievalDecision(
                "accepted",
                (HybridSearchResult(chunk, 0.3, vector_score=0.8, fts_score=1.0),),
                0.8,
                2,
            )
        return RetrievalDecision("no_relevant_information", (), None, 0)

    def model(**kwargs: object) -> AnswerResponse:
        blob = json.dumps(kwargs.get("messages"))
        if "kiosk specifications remain" in blob:
            return _draft(
                "The kiosk specifications remain.",
                [
                    Citation(
                        source_file=chunk.source_file,
                        chunk_id=chunk.id,
                        content_sha256=chunk.content_sha256 or "",
                        quote=note,
                    )
                ],
            )
        return _draft("No relevant information was found.", status="unknown")

    monkeypatch.setattr(agent, "retrieve_evidence", retrieve)
    monkeypatch.setattr(agent, "understand_query", _plan(None, "kiosk remaining work"))
    monkeypatch.setattr(agent, "call_llm_structured", model)

    response = agent.run_agent("what's left for the kiosk?")

    assert response.status == "answered"
    assert "kiosk specifications remain" in response.answer

