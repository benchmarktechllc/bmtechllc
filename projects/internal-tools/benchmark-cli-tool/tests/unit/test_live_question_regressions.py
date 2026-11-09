"""Regressions from live Ask BMTech questions, using synthetic chunks only.

The situations match questions people asked. The chunk text, channel names,
and card titles are invented for the test and are not copied from Discord,
the portal, or a meeting transcript.
"""

from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace

import pytest

from benchmark_cli import agent
from benchmark_cli.discord_ask import (
    DISCORD_MESSAGE_LIMIT,
    AskClient,
    AskConfig,
    format_discord_answer,
    thread_question_for,
)
from benchmark_cli.entailment import answers_question
from benchmark_cli.grounding import (
    asks_for_synced_channels,
    output_is_rejected,
    quote_covers_question,
)
from benchmark_cli.models import AnswerResponse, Citation
from benchmark_cli.retrieval import HybridSearchResult, RetrievalDecision
from benchmark_cli.storage.postgres import SearchChunk
from benchmark_cli.trello_answers import (
    OPEN_LISTS,
    TrelloCardRow,
    board_url_from_cards,
    requested_trello_list,
    trello_list_answer,
)

_DIGEST = "ab" * 32
_BOARD = "https://trello.com/b/tIvVZqio/bmtech-general-board"
_INVITE = "https://trello.com/invite/b/tIvVZqio/ATTIabcdefghijklmnopqrstuv/bmtech-general-board"
_FRAGMENT = 'So the sample portal and the chat feed are connected"].'
_GUIDE = "To add a page, open the editor, choose New page, and publish the draft."
_PORTAL = "The team is working on the smart kiosk project. The lobby pilot is underway."
_DISCORD = "Working notes: the smart kiosk project continues this month."
_TRELLO = "The team is working on the smart kiosk project named in this card."
_OTHER = "The cafeteria menu changes on Monday."
ASK_CHANNEL_ID = 100000000000000001


def _chunk(
    chunk_id: int,
    source: str,
    text: str,
    metadata: dict[str, object],
    *,
    digest: str | None = None,
) -> SearchChunk:
    return SearchChunk(
        chunk_id,
        source,
        0,
        text,
        metadata,
        1.0,
        content_sha256=digest or _DIGEST,
    )


def _citation(chunk: SearchChunk, quote: str | None = None) -> Citation:
    return Citation(
        source_file=chunk.source_file,
        chunk_id=chunk.id,
        content_sha256=chunk.content_sha256 or _DIGEST,
        quote=quote or chunk.chunk_text,
    )


def _row(
    list_name: str,
    title: str,
    card_id: str,
    chunk_id: int,
    *,
    chunk_text: str | None = None,
    url: str | None = None,
) -> TrelloCardRow:
    card_url = url or f"https://trello.com/c/{card_id}"
    text = chunk_text or f"Card: {title}\nList: {list_name}\nURL: {card_url}"
    return TrelloCardRow(
        board="Sample Board",
        list_name=list_name,
        card_id=card_id,
        title=title,
        url=card_url,
        chunk_id=chunk_id,
        source_file=f"trello/tIvVZqio/{card_id}.md",
        content_sha256=hashlib.sha256(card_id.encode()).hexdigest(),
        chunk_text=text,
    )


def _as_tuple(row: TrelloCardRow) -> tuple[object, ...]:
    return (
        row.board,
        row.list_name,
        row.card_id,
        row.title,
        row.url,
        row.chunk_id,
        row.source_file,
        row.content_sha256,
        row.chunk_text,
    )


@pytest.fixture
def quiet_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "classify_query", lambda _question: "cheap")
    monkeypatch.setattr(agent, "fts_search", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        agent,
        "call_llm",
        lambda **_kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=""))]
        ),
    )


def _run(
    monkeypatch: pytest.MonkeyPatch,
    question: str,
    chunks: tuple[SearchChunk, ...],
    claim: str,
    *,
    thread_question: str | None = None,
) -> AnswerResponse:
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision(
            "accepted",
            tuple(HybridSearchResult(chunk, 0.2, vector_score=0.8, fts_score=1.0) for chunk in chunks),
            0.8,
            1,
        ),
    )
    monkeypatch.setattr(
        agent,
        "call_llm",
        lambda **_kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=""))]
        ),
    )
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: AnswerResponse(
            status="answered",
            answer=claim,
            confidence=0.9,
            citations=[_citation(chunk) for chunk in chunks],
        ),
    )
    return agent.run_agent(question, thread_question=thread_question)


@pytest.mark.unit
def test_synced_channel_catalog_does_not_return_a_fragment(
    monkeypatch: pytest.MonkeyPatch, quiet_model: None
) -> None:
    question = "What channels has BMTech synced into the KB?"
    assert asks_for_synced_channels(question)
    assert asks_for_synced_channels("Which channels are being ingested?")
    assert asks_for_synced_channels("What channels are indexed?")
    assert output_is_rejected(_FRAGMENT)
    assert output_is_rejected('portal feed"].')
    assert output_is_rejected("the sentence already started.")
    monkeypatch.setattr(
        agent,
        "synced_discord_channel_rows",
        lambda: [
            ("sample-portal", 1, "discord-sample-portal.json", _DIGEST, "Synthetic portal note."),
            ("sample-planning", 2, "discord-sample-planning.json", _DIGEST, "Synthetic planning note."),
        ],
    )
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision("no_relevant_information", (), None, 0),
    )
    monkeypatch.setattr(agent, "call_llm", lambda **_kwargs: pytest.fail("model should not run"))
    monkeypatch.setattr(
        agent, "call_llm_structured", lambda **_kwargs: pytest.fail("model should not run")
    )

    response = agent.run_agent(question)

    assert response.status == "answered"
    assert response.answer == (
        "Synced Discord channels in the retrieval index: #sample-portal, #sample-planning."
    )
    assert _FRAGMENT not in response.answer
    assert "]. " not in response.answer
    assert output_is_rejected(response.answer) is False


@pytest.mark.unit
def test_add_page_howto_uses_the_page_identity(
    monkeypatch: pytest.MonkeyPatch, quiet_model: None
) -> None:
    guide = _chunk(
        4,
        "portal/guide/adding-a-page-from-the-portal.md",
        _GUIDE,
        {"source_type": "kb_page", "title": "Adding a page", "section": "Guide"},
    )
    assert answers_question("How do I add a page to the KB?", _GUIDE, require_all=True) is False
    assert quote_covers_question("How do I add a page to the KB?", _GUIDE, guide)

    response = _run(
        monkeypatch,
        "How do I add a page to the KB?",
        (guide,),
        "The retrieved evidence does not address this question.",
    )

    assert response.status == "answered"
    assert "publish the draft" in response.answer
    assert "does not address" not in response.answer.casefold()
    assert response.citations[0].source_file == guide.source_file


@pytest.mark.unit
def test_project_status_and_roster_questions_keep_covering_chunks(
    monkeypatch: pytest.MonkeyPatch, quiet_model: None
) -> None:
    portal = _chunk(
        11,
        "portal/projects/smart-kiosk.md",
        _PORTAL,
        {"source_type": "kb_page", "title": "Smart kiosk"},
        digest="1" * 64,
    )
    discord = _chunk(
        12,
        "discord-sample-projects.json",
        _DISCORD,
        {"source_type": "discord", "channel": "sample-projects"},
        digest="2" * 64,
    )
    trello = _chunk(
        13,
        "trello/tIvVZqio/kiosk.md",
        f"Card: Smart kiosk pilot\n{_TRELLO}",
        {"source_type": "trello", "title": "Smart kiosk pilot", "column": "Doing"},
        digest="3" * 64,
    )
    other = _chunk(
        14,
        "portal/office/cafeteria.md",
        _OTHER,
        {"source_type": "kb_page", "title": "Cafeteria"},
        digest="4" * 64,
    )
    chunks = (portal, discord, trello, other)

    status = _run(
        monkeypatch,
        "Current status of the smart kiosk project?",
        chunks,
        "The retrieved evidence does not address this question.",
    )
    projects = _run(
        monkeypatch,
        "What projects is BMTech working on?",
        chunks,
        "The retrieved evidence does not address this question.",
    )

    for response in (status, projects):
        assert response.status == "answered"
        assert "smart kiosk" in response.answer.casefold()
        assert "cafeteria" not in response.answer.casefold()
        assert "does not address" not in response.answer.casefold()
        assert {citation.chunk_id for citation in response.citations} == {11, 12, 13}


@pytest.mark.unit
def test_open_trello_tickets_list_every_list_except_done(
    monkeypatch: pytest.MonkeyPatch, quiet_model: None
) -> None:
    rows = (
        _row("Backlog", "Sample backlog card", "back", 1),
        _row("Doing", "Sample doing card", "doing", 2),
        _row("Review", "Sample review card", "review", 3),
        _row("Done", "Sample done card", "done", 4),
    )
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [_as_tuple(row) for row in rows])
    monkeypatch.setattr(agent, "retrieve_evidence", lambda *_a, **_k: pytest.fail("retrieval"))
    monkeypatch.setattr(agent, "call_llm", lambda **_k: pytest.fail("model"))

    assert requested_trello_list("Which Trello tickets are still open?") == OPEN_LISTS
    assert requested_trello_list("Which tickets are not done?") == OPEN_LISTS
    assert requested_trello_list("Which Trello cards are open?") == OPEN_LISTS

    response = agent.run_agent("Which Trello tickets are still open?")

    assert response.status == "answered"
    assert "Trello cards in Backlog:" in response.answer
    assert "Trello cards in Doing:" in response.answer
    assert "Trello cards in Review:" in response.answer
    assert "Trello cards in Done:" not in response.answer
    assert "Sample done card" not in response.answer
    assert "Sample backlog card" in response.answer
    assert "Sample doing card" in response.answer
    assert "Sample review card" in response.answer


@pytest.mark.unit
def test_trello_board_link_is_derived_and_not_an_invite(
    monkeypatch: pytest.MonkeyPatch, quiet_model: None
) -> None:
    row = _row(
        "Doing",
        "Sample card",
        "board",
        5,
        chunk_text=(
            "Card: Sample card\n"
            f"Board URL: {_BOARD}\n"
            f"Invite: {_INVITE}\n"
        ),
    )
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [_as_tuple(row)])
    monkeypatch.setattr(agent, "retrieve_evidence", lambda *_a, **_k: pytest.fail("retrieval"))
    monkeypatch.setattr(agent, "call_llm", lambda **_k: pytest.fail("model"))

    response = agent.run_agent("link our trello board")
    rendered = response.model_dump_json()

    assert response.status == "answered"
    assert response.answer == f"The Trello board is {_BOARD}."
    assert "ATTI" not in rendered
    assert "/invite/" not in rendered
    assert _BOARD in response.citations[0].quote


@pytest.mark.unit
def test_invite_only_rows_become_the_canonical_board_url() -> None:
    row = _row(
        "Backlog",
        "Sample card",
        "invite",
        6,
        chunk_text=f"Card: Sample card\nInvite: {_INVITE}\n",
    )

    assert board_url_from_cards((row,)) == _BOARD


@pytest.mark.unit
def test_backlog_list_fits_eleven_cards_and_caps_a_longer_list() -> None:
    long_title = "Sample backlog item " + ("x" * 60)
    eleven = [
        _row("Backlog", f"{long_title} {index:02d}", f"long{index}", index + 1)
        for index in range(11)
    ]
    eleven_response = trello_list_answer("backlog", eleven)
    rendered = format_discord_answer(eleven_response)
    assert len(rendered) <= DISCORD_MESSAGE_LIMIT
    assert "**Sources**" in rendered
    assert "**Citations**" not in rendered
    for row in eleven:
        assert row.title in rendered

    wide = [
        _row("Backlog", f"{long_title} wide {index:02d}", f"wide{index}", index + 1)
        for index in range(25)
    ]
    wide_response = trello_list_answer("backlog", wide)
    wide_rendered = format_discord_answer(wide_response)
    assert len(wide_rendered) <= DISCORD_MESSAGE_LIMIT
    assert "…and " in wide_response.answer
    assert wide_response.answer.endswith("more.")
    assert len(wide_response.citations) < 25


@pytest.mark.unit
def test_bare_list_follow_up_matches_an_indexed_list(
    monkeypatch: pytest.MonkeyPatch, quiet_model: None
) -> None:
    doing = _row("Doing", "Sample doing card", "doing", 2)
    backlog = _row("Backlog", "Sample backlog card", "back", 1)
    monkeypatch.setattr(
        agent, "trello_card_rows", lambda: [_as_tuple(doing), _as_tuple(backlog)]
    )
    monkeypatch.setattr(agent, "call_llm", lambda **_kwargs: pytest.fail("model"))
    monkeypatch.setattr(
        agent, "retrieve_evidence", lambda *_a, **_k: pytest.fail("retrieval")
    )

    response = agent.run_agent(
        "In doing state?",
        thread_question="Which Trello tickets are still open?",
    )

    assert response.status == "answered"
    assert "Sample doing card" in response.answer
    assert "Sample backlog card" not in response.answer
    assert thread_question_for(
        "In doing state?", "Which Trello tickets are still open?"
    ) == "Which Trello tickets are still open?"


@pytest.mark.unit
def test_bare_follow_up_without_that_list_keeps_the_thread_question(
    monkeypatch: pytest.MonkeyPatch, quiet_model: None
) -> None:
    backlog = _row("Backlog", "Sample backlog card", "back", 1)
    seen: dict[str, str] = {}

    def retrieve(question: str, **_kwargs: object) -> RetrievalDecision:
        seen["question"] = question
        return RetrievalDecision("no_relevant_information", (), None, 0)

    monkeypatch.setattr(agent, "trello_card_rows", lambda: [_as_tuple(backlog)])
    monkeypatch.setattr(agent, "retrieve_evidence", retrieve)

    response = agent.run_agent(
        "In doing state?",
        thread_question="What is the sample project status?",
    )

    assert response.status == "no_relevant_information"
    assert "Sample backlog card" not in response.answer
    assert "What is the sample project status?" in seen["question"]
    assert "In doing state?" in seen["question"]


@pytest.mark.unit
def test_thread_starter_is_passed_into_the_agent() -> None:
    seen: dict[str, str | None] = {}

    def runner(question: str, thread_question: str | None = None, **_kwargs: object) -> AnswerResponse:
        seen["question"] = question
        seen["thread"] = thread_question
        return AnswerResponse(
            status="no_relevant_information",
            answer="No relevant information was found in the company document library.",
            confidence=0.0,
        )

    client = AskClient(AskConfig(ASK_CHANNEL_ID, None), agent=runner)
    rendered = asyncio.run(
        client.answer_question(
            "In doing state?",
            thread_question="Which Trello tickets are still open?",
        )
    )

    assert seen == {
        "question": "In doing state?",
        "thread": "Which Trello tickets are still open?",
    }
    assert "Citations" not in rendered


@pytest.mark.unit
def test_headcount_and_missing_certificate_stay_unknown(
    monkeypatch: pytest.MonkeyPatch, quiet_model: None
) -> None:
    portal = _chunk(
        21,
        "portal/projects/smart-kiosk.md",
        _PORTAL,
        {"source_type": "kb_page", "title": "Smart kiosk"},
    )
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision(
            "accepted",
            (HybridSearchResult(portal, 0.2, vector_score=0.8, fts_score=1.0),),
            0.8,
            1,
        ),
    )
    monkeypatch.setattr(agent, "call_llm", lambda **_kwargs: pytest.fail("headcount must not call the model"))
    monkeypatch.setattr(
        agent, "call_llm_structured", lambda **_kwargs: pytest.fail("headcount must not call the model")
    )

    headcount = agent.run_agent("How many employees does BMTech have?")

    assert headcount.status == "no_relevant_information"
    assert "does not state a current employee count" in headcount.answer.casefold()
    assert "smart kiosk" not in headcount.answer.casefold()
    assert "employee" in headcount.answer.casefold()
    assert not __import__("re").search(r"\b\d+\s+employees?\b", headcount.answer)

    notes = _chunk(
        22,
        "portal/notes/onboarding.md",
        "Internal notes cover the sample onboarding checklist.",
        {"source_type": "kb_page", "title": "Onboarding"},
        digest="c" * 64,
    )
    certificate = _run(
        monkeypatch,
        "Link to the Anthropic/Claude agentic AI certificate?",
        (notes,),
        "The certificate is at https://example.test/claude-agent-certificate.",
    )

    assert certificate.status == "no_relevant_information"
    assert certificate.citations == []
    assert "example.test" not in certificate.answer
    assert "anthropic" not in certificate.answer.casefold()
    assert "claude" not in certificate.answer.casefold()
    assert "does not address" in certificate.answer.casefold()


_STEPS = (
    "1. Open the section the page belongs in.",
    "2. Choose the sample editor.",
    "3. Fill in the form (Section, Title, Description, Owner, Markdown).",
    "4. Write a sample paragraph.",
    "5. Set the sample owner.",
    "6. Check the sample preview.",
    "7. Publish the sample draft.",
)


@pytest.mark.unit
def test_board_link_uses_another_chunk_when_cards_only_have_card_urls(
    monkeypatch: pytest.MonkeyPatch, quiet_model: None
) -> None:
    row = _row("Doing", "Sample card", "card", 8, url="https://trello.com/c/card")
    note = "Board link - https://trello.com/b/tIvVZqio/bmtech-general-board"
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [_as_tuple(row)])
    monkeypatch.setattr(
        agent,
        "board_url_chunk_rows",
        lambda limit=8: [(90, "discord-sample.json", "d" * 64, note)],
    )
    monkeypatch.setattr(agent, "retrieve_evidence", lambda *_a, **_k: pytest.fail("retrieval"))
    monkeypatch.setattr(agent, "call_llm", lambda **_k: pytest.fail("model"))

    response = agent.run_agent("link our trello board")
    rendered = response.model_dump_json()

    assert response.status == "answered"
    assert response.answer == f"The Trello board is {_BOARD}."
    assert response.citations[0].quote == note
    assert "trello/tIvVZqio/" in row.source_file
    assert "ATTI" not in rendered
    assert "/invite/" not in rendered
    assert "https://trello.com/c/" not in response.answer


@pytest.mark.unit
def test_projects_question_returns_excerpts_without_work_or_bmtech_words(
    monkeypatch: pytest.MonkeyPatch, quiet_model: None
) -> None:
    portal = _chunk(
        31,
        "portal/projects/sample-portal.md",
        "The sample portal project is in a pilot.",
        {"source_type": "kb_page", "title": "Sample portal"},
        digest="1" * 64,
    )
    discord = _chunk(
        32,
        "discord-sample-projects.json",
        "Notes: the sample kiosk project continues.",
        {"source_type": "discord", "channel": "sample-projects"},
        digest="2" * 64,
    )
    other = _chunk(
        33,
        "portal/office/cafeteria.md",
        "The cafeteria menu changes on Monday.",
        {"source_type": "kb_page", "title": "Cafeteria"},
        digest="3" * 64,
    )
    for text in (portal.chunk_text, discord.chunk_text, other.chunk_text):
        lowered = text.casefold()
        assert "work" not in lowered
        assert "bmtech" not in lowered
        assert "currently" not in lowered

    response = _run(
        monkeypatch,
        "What projects is BMTech working on?",
        (portal, discord, other),
        "The retrieved evidence does not address this question.",
    )

    assert response.status == "answered"
    assert "sample portal project" in response.answer.casefold()
    assert "sample kiosk project" in response.answer.casefold()
    assert "cafeteria" not in response.answer.casefold()
    assert "does not address" not in response.answer.casefold()


@pytest.mark.unit
def test_add_page_howto_keeps_every_numbered_step(
    monkeypatch: pytest.MonkeyPatch, quiet_model: None
) -> None:
    guide = _chunk(
        41,
        "portal/guide/adding-a-page-from-the-portal.md",
        "\n".join(_STEPS),
        {"source_type": "kb_page", "title": "Adding a page", "section": "Guide"},
    )
    response = _run(
        monkeypatch,
        "How do I add a page to the KB?",
        (guide,),
        "Choose the sample editor. Write a sample paragraph. Set the sample owner. Check the sample preview.",
    )

    assert response.status == "answered"
    for step in _STEPS:
        assert step in response.answer
    assert "does not address" not in response.answer.casefold()


@pytest.mark.unit
def test_status_excerpts_drop_extra_list_markers_and_join_card_headers(
    monkeypatch: pytest.MonkeyPatch, quiet_model: None
) -> None:
    note = _chunk(
        51,
        "portal/projects/sample-kiosk.md",
        "- MVP definition for the smart kiosk project.",
        {"source_type": "kb_page", "title": "Sample kiosk"},
        digest="a" * 64,
    )
    card = _chunk(
        52,
        "trello/tIvVZqio/kiosk.md",
        "Card: Sample kiosk\nBoard: Sample Board\nList: Doing\n- MVP definition for the smart kiosk project.",
        {"source_type": "trello", "title": "Sample kiosk", "column": "Doing"},
        digest="b" * 64,
    )
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [])
    response = _run(
        monkeypatch,
        "Current status of the smart kiosk project?",
        (note, card),
        "The retrieved evidence does not address this question.",
    )

    assert response.status == "answered"
    assert "- -" not in response.answer
    assert "MVP definition for the smart kiosk project." in response.answer
    assert "Board: Sample Board / List: Doing" in response.answer


@pytest.mark.unit
def test_open_trello_answer_prefixes_a_per_list_count(
    monkeypatch: pytest.MonkeyPatch, quiet_model: None
) -> None:
    rows = [
        _row("Backlog", f"Sample backlog {index}", f"b{index}", index + 1) for index in range(2)
    ]
    rows.append(_row("Doing", "Sample doing card", "doing", 20))
    rows.extend(_row("Sev 1", f"Sample sev {index}", f"s{index}", 30 + index) for index in range(3))
    rows.append(_row("Done", "Sample done card", "done", 40))
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [_as_tuple(row) for row in rows])
    monkeypatch.setattr(agent, "retrieve_evidence", lambda *_a, **_k: pytest.fail("retrieval"))
    monkeypatch.setattr(agent, "call_llm", lambda **_k: pytest.fail("model"))

    response = agent.run_agent("Which Trello tickets are still open?")
    rendered = format_discord_answer(response)

    assert response.answer.splitlines()[0] == "Backlog 2, Doing 1, Sev 1 3"
    assert "Done" not in response.answer.splitlines()[0]
    assert "Sample done card" not in response.answer
    assert len(rendered) <= DISCORD_MESSAGE_LIMIT


_PAGE = """\
1. Open the section the page belongs in, e.g.
Guide or Projects.
2. Choose New page in the sample editor.
3. Fill in the form:
| Field | Value |
| --- | --- |
| Section | Guide |
| Title | Page name |
| Description | What the page covers |
| Owner | The author |
| Markdown | The page body |
4. Watch the sample preview panel as you type.
It shows the exact file path.
5. Save the sample draft.
6. Ask a teammate to read the sample page.
7. New pages are created as `status: draft`, internal, untagged. Add tags before review.

> [!NOTE]
> Raw HTML is rejected.
> Keep the sample body in Markdown.
"""


@pytest.mark.unit
def test_add_page_procedure_keeps_examples_and_table_fields(
    monkeypatch: pytest.MonkeyPatch, quiet_model: None
) -> None:
    guide = _chunk(
        61,
        "portal/guide/adding-a-page-from-the-portal.md",
        _PAGE,
        {"source_type": "kb_page", "title": "Adding a page", "section": "Guide"},
    )
    short_quote = _PAGE[:120]
    assert short_quote.rstrip().endswith("e.g.") or "e.g." in short_quote
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_a, **_k: RetrievalDecision(
            "accepted",
            (HybridSearchResult(guide, 0.2, vector_score=0.8, fts_score=1.0),),
            0.8,
            1,
        ),
    )
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_k: AnswerResponse(
            status="answered",
            answer="1. Open the section the page belongs in, e.g.\n3. Fill in the form:\n4. Watch the sample preview panel as you type.",
            confidence=0.9,
            citations=[_citation(guide, short_quote)],
        ),
    )

    response = agent.run_agent("How do I add a page to the KB?")

    assert response.status == "answered"
    assert "e.g. Guide or Projects." in response.answer
    assert "Section: Guide" in response.answer
    assert "Title: Page name" in response.answer
    assert "Description: What the page covers" in response.answer
    assert "Owner: The author" in response.answer
    assert "Markdown: The page body" in response.answer
    for number in range(1, 8):
        assert f"{number}." in response.answer
    for line in response.answer.splitlines():
        body = line.split(".", 1)[-1].strip()
        assert not body.endswith((":", "e.g.", "e.g"))
    assert "exact file path." in response.answer
    assert "`status: draft`, internal, untagged" in response.answer
    assert "` ." not in response.answer
    assert response.answer.count("`") % 2 == 0
    assert "[!NOTE]" not in response.answer
    assert " > " not in response.answer
    assert "Note: Raw HTML is rejected." in response.answer


@pytest.mark.unit
def test_project_status_reports_the_matching_card_list_and_link(
    monkeypatch: pytest.MonkeyPatch, quiet_model: None
) -> None:
    title = "MVP definition and setting a timeline for the smart kiosk and cli-tool projects"
    card = _row("Doing", title, "mvp", 70)
    other = _row("Backlog", "Sample cafeteria refresh", "cafe", 71)
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [_as_tuple(card), _as_tuple(other)])
    monkeypatch.setattr(agent, "call_llm", lambda **_k: pytest.fail("model"))
    monkeypatch.setattr(agent, "retrieve_evidence", lambda *_a, **_k: pytest.fail("retrieval"))

    response = agent.run_agent("Current status of the smart kiosk project?")

    assert response.status == "answered"
    assert title in response.answer
    assert "Doing" in response.answer
    assert "https://trello.com/c/mvp" in response.answer
    assert "cafeteria" not in response.answer.casefold()
    assert "does not address" not in response.answer.casefold()
    assert "- -" not in response.answer


@pytest.mark.unit
def test_working_on_projects_lists_doing_as_active_and_backlog_as_planned(
    monkeypatch: pytest.MonkeyPatch, quiet_model: None
) -> None:
    doing_bot = _row(
        "Doing",
        "Discord bot hooked into RAG",
        "bot",
        82,
        chunk_text=(
            "Card: Discord bot hooked into RAG\n"
            "List: Doing\n"
            "URL: https://trello.com/c/bot\n"
            "\n"
            "The bot answers company questions from retrieved notes."
        ),
    )
    doing_kiosk = _row(
        "Doing",
        "MVP definition for the smart kiosk and cli-tool projects",
        "mvp2",
        83,
    )
    planned = _row("Backlog", "Phase 2 portfolio projects", "phase", 84)
    chore = _row("Backlog", "Order sticky notes", "notes", 85)
    monkeypatch.setattr(
        agent,
        "trello_card_rows",
        lambda: [_as_tuple(card) for card in (doing_bot, doing_kiosk, planned, chore)],
    )
    monkeypatch.setattr(agent, "call_llm", lambda **_k: pytest.fail("model"))
    monkeypatch.setattr(agent, "retrieve_evidence", lambda *_a, **_k: pytest.fail("retrieval"))

    response = agent.run_agent("What projects is BMTech working on?")

    assert response.status == "answered"
    active_at = response.answer.index("Active work:")
    planned_at = response.answer.index("Backlog (planned), not active:")
    assert active_at < planned_at
    assert "Discord bot hooked into RAG" in response.answer
    assert "smart kiosk and cli-tool" in response.answer
    assert "Doing (active)" in response.answer
    assert "Phase 2 portfolio projects" in response.answer
    assert "Backlog (planned)" in response.answer
    assert "not active" in response.answer.casefold()
    assert "sticky" not in response.answer.casefold()
    assert "current" not in response.answer.casefold()
    assert "- Card:" not in response.answer
    cited = {citation.quote for citation in response.citations}
    assert "Card: Discord bot hooked into RAG" in cited
    assert "Card: Phase 2 portfolio projects" in cited
    assert "Card: Order sticky notes" not in cited


@pytest.mark.unit
def test_open_list_footer_counts_shown_cards_against_the_summary() -> None:
    from benchmark_cli.models import Citation

    summary = "Backlog 11, Doing 4, Sev 1 12, Sev 2 2, Sev 3 3"
    citations = [
        Citation(
            source_file=f"trello/tIvVZqio/c{index}.md",
            chunk_id=index + 1,
            content_sha256="ab" * 32,
            quote=f"Card: Sample {index}",
        )
        for index in range(20)
    ]
    rendered = format_discord_answer(
        AnswerResponse(
            status="answered",
            answer=f"{summary}\n\nTrello cards in Backlog:\n- [Sample](https://trello.com/c/sample)",
            confidence=0.8,
            citations=citations,
        )
    )

    assert "20 shown of 32" in rendered
