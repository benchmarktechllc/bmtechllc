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
def test_board_typo_does_not_route_to_one_person(monkeypatch: pytest.MonkeyPatch) -> None:
    _quiet(monkeypatch)
    monkeypatch.setattr(
        agent,
        "understand_query",
        _plan("Dante Paniccia", "what's on the trello board"),
    )
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: _draft("No relevant information was found.", status="unknown"),
    )

    response = agent.run_agent("what's on the trelo board")

    assert "Ship the portal" in response.answer
    assert "Kiosk MVP" in response.answer
    assert "Dante Paniccia's tickets" not in response.answer


@pytest.mark.unit
def test_named_item_status_does_not_dump_the_done_list(monkeypatch: pytest.MonkeyPatch) -> None:
    _quiet(monkeypatch)
    monkeypatch.setattr(agent, "understand_query", _plan(None, "lobby copy status"))
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: _draft("No relevant information was found.", status="unknown"),
    )

    response = agent.run_agent("is the lobby copy done yet?")

    assert response.status == "answered"
    assert "Review the lobby copy" in response.answer
    assert "not done" in response.answer.casefold()
    assert "Old portal import" not in response.answer


@pytest.mark.unit
def test_completed_list_is_in_the_body_and_citations_match_that_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _quiet(monkeypatch)
    ship = sample_board()[0]
    monkeypatch.setattr(agent, "understand_query", _plan(None, "completed board"))
    answer = "Several tasks have been completed on the board, listed under the Done section."
    citation = Citation(
        source_file=ship.source_file,
        chunk_id=ship.chunk_id,
        content_sha256=ship.content_sha256,
        quote="Card: Ship the portal",
    )
    monkeypatch.setattr(agent, "call_llm_structured", lambda **_kwargs: _draft(answer, [citation]))

    response = agent.run_agent("What has been completed on the board?")

    assert "Old portal import" in response.answer
    assert "Retired kiosk notes" in response.answer
    assert "Ship the portal" not in response.answer
    cited = {item.quote for item in response.citations}
    assert "Card: Ship the portal" not in cited
    assert "Card: Old portal import" in cited


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


@pytest.mark.unit
def test_team_question_uses_the_roster(monkeypatch: pytest.MonkeyPatch) -> None:
    _quiet(monkeypatch)
    monkeypatch.setattr(agent, "understand_query", _plan(None, "team roster"))
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: _draft("No relevant information was found.", status="unknown"),
    )

    response = agent.run_agent("who's on the team?")

    assert response.status == "answered"
    assert "Dante Paniccia" in response.answer
    assert "Fanuel Mengistu" in response.answer
    assert "Anwar Hussein" in response.answer


@pytest.mark.unit
def test_compound_channel_and_list_answers_both_halves(monkeypatch: pytest.MonkeyPatch) -> None:
    _quiet(monkeypatch)
    monkeypatch.setattr(agent, "trello_card_rows", lambda: _sev_board())
    monkeypatch.setattr(
        agent,
        "synced_discord_channel_rows",
        lambda: [("sample-portal", 1, "discord-sample.json", "ab" * 32, "Synthetic portal note.")],
    )
    monkeypatch.setattr(agent, "understand_query", _plan(None, "channels and sev 2"))
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: _draft("No relevant information was found.", status="unknown"),
    )

    response = agent.run_agent("which channels does the bot read and what's in Sev 2?")

    assert "sample-portal" in response.answer
    assert "Sample sev two note" in response.answer


@pytest.mark.unit
def test_sev_definition_is_unknown_and_may_name_the_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    _quiet(monkeypatch)
    monkeypatch.setattr(agent, "trello_card_rows", lambda: _sev_board())
    monkeypatch.setattr(agent, "understand_query", _plan(None, "sev 2 definition"))
    sev = _sev_board()[-1]
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: _draft(
            f"Trello cards in Sev 2:\n- [{sev.title}]({sev.url})",
            [
                Citation(
                    source_file=sev.source_file,
                    chunk_id=sev.chunk_id,
                    content_sha256=sev.content_sha256,
                    quote="Card: Sample sev two note",
                )
            ],
        ),
    )

    response = agent.run_agent("what does Sev 2 mean?")

    assert response.status == "no_relevant_information"
    assert "no definition" in response.answer.casefold()
    assert "Sample sev two note" in response.answer
    assert response.citations == []


@pytest.mark.unit
def test_current_person_work_labels_completed_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    _quiet(monkeypatch)
    monkeypatch.setattr(agent, "understand_query", _plan("Fanuel Mengistu", "Fanuel Mengistu"))
    kiosk = sample_board()[2]
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: _draft(
            "Fanuel Mengistu is working on [Kiosk MVP](https://trello.com/c/kiosk).",
            [
                Citation(
                    source_file=kiosk.source_file,
                    chunk_id=kiosk.chunk_id,
                    content_sha256=kiosk.content_sha256,
                    quote="Card: Kiosk MVP",
                )
            ],
        ),
    )

    response = agent.run_agent("What is Fanuel currently working on")

    assert "Kiosk MVP" in response.answer
    assert "Retired kiosk notes" in response.answer
    assert "Completed:" in response.answer
    assert response.answer.index("Kiosk MVP") < response.answer.index("Retired kiosk notes")


@pytest.mark.unit
def test_missing_list_beats_unrelated_process_advice(monkeypatch: pytest.MonkeyPatch) -> None:
    _quiet(monkeypatch)
    advice = (
        "The QA process should involve establishing objective quality criteria "
        "and test environments before execution begins."
    )
    note = SearchChunk(
        70,
        "portal/guide/sample-plan.md",
        0,
        advice,
        {"source_type": "kb_page", "title": "Sample plan"},
        0.6,
        content_sha256=hashlib.sha256(b"qa-advice").hexdigest(),
    )
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision(
            "accepted",
            (HybridSearchResult(note, 0.2, vector_score=0.6, fts_score=1.0),),
            0.6,
            2,
        ),
    )
    monkeypatch.setattr(agent, "understand_query", _plan(None, "QA process"))
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: _draft(
            advice,
            [
                Citation(
                    source_file=note.source_file,
                    chunk_id=note.id,
                    content_sha256=note.content_sha256 or "",
                    quote=advice,
                )
            ],
        ),
    )

    response = agent.run_agent("What's in QA?")

    assert response.status == "no_relevant_information"
    assert 'named "qa"' in response.answer.casefold()
    assert "test environments" not in response.answer.casefold()


@pytest.mark.unit
def test_board_link_and_urgent_are_both_answered(monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmark_cli.trello_answers import TrelloCardRow

    _quiet(monkeypatch)
    rows = list(sample_board())
    ship = rows[0]
    board_url = "https://trello.com/b/tIvVZqio/sample-board"
    rows[0] = TrelloCardRow(
        board=ship.board,
        list_name=ship.list_name,
        card_id=ship.card_id,
        title=ship.title,
        url=ship.url,
        chunk_id=ship.chunk_id,
        source_file=ship.source_file,
        content_sha256=ship.content_sha256,
        chunk_text=ship.chunk_text + f"\nBoard URL: {board_url}",
    )
    rows.append(
        TrelloCardRow(
            board="Sample Board",
            list_name="Sev 1",
            card_id="sev1",
            title="Sample urgent outage",
            url="https://trello.com/c/sev1",
            chunk_id=8,
            source_file="trello/tIvVZqio/sev1.md",
            content_sha256=hashlib.sha256(b"sev1").hexdigest(),
            chunk_text="Card: Sample urgent outage\nList: Sev 1\nURL: https://trello.com/c/sev1",
        )
    )
    monkeypatch.setattr(agent, "trello_card_rows", lambda: rows)
    monkeypatch.setattr(agent, "understand_query", _plan(None, "trello board urgent"))
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: _draft("Most urgent tickets are Sev 1.", status="unknown"),
    )

    response = agent.run_agent("where do I find the Trello board and what's most urgent on it?")

    assert board_url in response.answer
    assert "Sample urgent outage" in response.answer

