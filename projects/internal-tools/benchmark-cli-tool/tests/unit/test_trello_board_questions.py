"""Paraphrases that must be answered from the whole indexed board."""

from __future__ import annotations

import pytest

from benchmark_cli import agent
from benchmark_cli.discord_ask import format_discord_messages
from benchmark_cli.models import AnswerResponse, Citation
from benchmark_cli.trello_answers import answer_matches_snapshot, prefer_validated_draft

_LISTS = (
    ("Backlog", 11),
    ("Doing", 4),
    ("Done", 12),
    ("Sev 1", 12),
    ("Sev 2", 2),
    ("Sev 3", 3),
)


def _board(monkeypatch: pytest.MonkeyPatch):
    from tests.unit.test_live_question_regressions import _as_tuple, _row

    rows = []
    chunk_id = 1
    for list_name, count in _LISTS:
        for index in range(count):
            title = f"Sample {list_name} ticket number {index + 1:02d} for the board catalog"
            description = ""
            if list_name == "Doing" and index == 0:
                title = "Sample knowledge base guide"
                description = "Track the knowledge base guide from this card."
            elif list_name == "Done" and index == 0:
                title = "Finished knowledge base import"
                description = "The knowledge base import is already finished."
            elif list_name == "Backlog" and index == 0:
                title = "Sample MCP project"
                description = "Notes for the sample MCP project card."
            text = f"Card: {title}\nList: {list_name}\nURL: https://trello.com/c/c{chunk_id}"
            if description:
                text = f"{text}\n\n{description}"
            rows.append(
                _row(list_name, title, f"c{chunk_id}", chunk_id, chunk_text=text)
            )
            chunk_id += 1
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [_as_tuple(row) for row in rows])
    monkeypatch.setattr(agent, "call_llm", lambda **_kwargs: pytest.fail("model"))
    monkeypatch.setattr(agent, "retrieve_evidence", lambda *_args, **_kwargs: pytest.fail("retrieval"))
    return rows


@pytest.fixture
def quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)


@pytest.mark.unit
def test_doing_right_now_lists_the_four_doing_cards(monkeypatch: pytest.MonkeyPatch, quiet: None) -> None:
    _board(monkeypatch)

    response = agent.run_agent("What Trello cards are in Doing right now?")

    assert response.status == "answered"
    assert "doing right now" not in response.answer.casefold()
    assert response.answer.count("Sample Doing ticket") == 3
    assert "Sample knowledge base guide" in response.answer
    assert "https://trello.com/c/" in response.answer


@pytest.mark.unit
def test_open_tickets_list_every_non_done_card_across_discord_messages(
    monkeypatch: pytest.MonkeyPatch, quiet: None
) -> None:
    rows = _board(monkeypatch)

    response = agent.run_agent("Which tickets are still open?")
    parts = format_discord_messages(response)
    combined = "\n".join(parts)

    assert all(len(part) <= 2000 for part in parts)
    assert len(parts) > 1
    assert "Backlog 11" in response.answer
    assert "Doing 4" in response.answer
    assert "Sev 1 12" in response.answer
    assert "Sev 2 2" in response.answer
    assert "Sev 3 3" in response.answer
    assert "Done 12" not in response.answer.splitlines()[0]
    for row in rows:
        if row.list_name == "Done":
            assert row.title not in combined
        else:
            assert row.title in combined
    assert "…and " not in response.answer


@pytest.mark.unit
@pytest.mark.parametrize(
    "question",
    (
        "Show me the Backlog tickets",
        "What's in Backlog?",
        "Backlog tickets?",
        "How many cards are in Backlog?",
    ),
)
def test_backlog_paraphrases_list_eleven_cards(
    monkeypatch: pytest.MonkeyPatch, quiet: None, question: str
) -> None:
    _board(monkeypatch)

    response = agent.run_agent(question, thread_question="What Trello cards are in Doing right now?")

    assert response.answer.count("Sample Backlog ticket") == 10
    assert "Sample MCP project" in response.answer
    assert "Sample Doing ticket" not in response.answer


@pytest.mark.unit
def test_remaining_board_paraphrases(monkeypatch: pytest.MonkeyPatch, quiet: None) -> None:
    rows = _board(monkeypatch)

    sev = agent.run_agent("List the Sev 1 cards")
    assert sev.answer.count("Sample Sev 1 ticket") == 12

    done = agent.run_agent("Which tickets are done?")
    completed = agent.run_agent("What has been completed on the board?")
    assert done.answer.count("Sample Done ticket") == 11
    assert "Finished knowledge base import" in done.answer
    assert completed.answer.count("Sample Done ticket") == 11

    progress = agent.run_agent("What's currently in progress on Trello?")
    doing = agent.run_agent("Which cards are in the Doing list currently?")
    assert progress.answer.count("Sample Doing ticket") == 3
    assert "Sample knowledge base guide" in progress.answer
    assert doing.answer.count("Sample Doing ticket") == 3

    missing = agent.run_agent("What's in QA?")
    assert 'named "qa"' in missing.answer.casefold()
    assert "Backlog" in missing.answer
    assert "Doing" in missing.answer
    for row in rows:
        assert row.title not in missing.answer

    detail = agent.run_agent("Tell me about the MCP project card")
    assert "Sample MCP project" in detail.answer
    assert "Backlog" in detail.answer
    assert "Notes for the sample MCP project card." in detail.answer
    assert "https://trello.com/c/" in detail.answer

    assigned = agent.run_agent("Which tickets are assigned to Fanuel?")
    assert "not indexed" in assigned.answer.casefold()
    assert "Fanuel" not in assigned.answer

    due = agent.run_agent("Any tickets due this week?")
    assert "do not have a due date" in due.answer.casefold() or "none of the indexed" in due.answer.casefold()

    topical = agent.run_agent("Any open tickets about the knowledge base?")
    assert "Sample knowledge base guide" in topical.answer
    assert "Finished knowledge base import" not in topical.answer
    assert "Sample Doing ticket" not in topical.answer


@pytest.mark.unit
def test_fabricated_snapshot_draft_falls_back(monkeypatch: pytest.MonkeyPatch, quiet: None) -> None:
    from tests.unit.test_live_question_regressions import _as_tuple, _row

    rows = (
        _row("Doing", "Sample Doing card", "d1", 1),
        _row("Backlog", "Sample Backlog card", "b1", 2),
    )
    citation = Citation(
        source_file="trello/tIvVZqio/d1.md",
        chunk_id=1,
        content_sha256=rows[0].content_sha256,
        quote="Card: Sample Doing card",
    )
    draft = AnswerResponse(
        status="answered",
        answer="Doing has [Invented card](https://trello.com/c/nope).",
        confidence=0.9,
        citations=[citation],
    )

    assert answer_matches_snapshot(draft.answer, rows) is False
    kept = prefer_validated_draft(draft, rows, AnswerResponse(
        status="answered",
        answer="- [Sample Doing card](https://trello.com/c/d1) — Doing",
        confidence=0.8,
        citations=[citation],
    ))
    assert "Invented card" not in kept.answer
    assert "Sample Doing card" in kept.answer
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [_as_tuple(row) for row in rows])
