"""Typo-tolerant roster matching. No model calls and no database."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from benchmark_cli.names import resolve_question
from benchmark_cli.trello_answers import TrelloCardRow, answer_board_question
from benchmark_cli.trello_sync import TrelloBoard, TrelloCard, units_for_card

_DIGEST = "ab" * 32


def _card(card_id: str, title: str, members: str) -> TrelloCardRow:
    return TrelloCardRow(
        "Sample Board",
        "Doing",
        card_id,
        title,
        f"https://trello.com/c/{card_id}",
        1 if card_id == "audit" else 2,
        f"trello/board/{card_id}.md",
        _DIGEST,
        f"Card: {title}\nList: Doing\nMembers: {members}\nURL: https://trello.com/c/{card_id}",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("question", "canonical"),
    (
        ("What tickets are assigned to Dante", "Dante Paniccia"),
        ("what tickets are assigned to dante", "Dante Paniccia"),
        ("Tickets assigned to Dantino", "Dante Paniccia"),
        ("Tickets assigned to Fanual", "Fanuel Mengistu"),
        ("Which tickets are assigned to Fanuel?", "Fanuel Mengistu"),
        ("Tickets assigned to Fanuell", "Fanuel Mengistu"),
        ("Tickets assigned to Fano", "Fanuel Mengistu"),
        ("Tickets assigned to Anwr", "Anwar Hussein"),
        ("Tickets assigned to Anwar", "Anwar Hussein"),
        ("what is the kisok status", "kiosk"),
        ("what is Trelo", "Trello"),
    ),
)
def test_fuzzy_names_resolve_people_and_terms(question: str, canonical: str) -> None:
    resolution = resolve_question(question)
    assert resolution.changed
    assert canonical in resolution.query
    assert canonical in resolution.note
    assert question.split()[0] in resolution.query or question.casefold().startswith("what")


@pytest.mark.unit
def test_questions_without_names_are_unchanged() -> None:
    question = "How many employees does BMTech have?"
    resolution = resolve_question(question)
    assert resolution.changed is False
    assert resolution.query == question
    assert resolution.note == ""
    assert resolution.people == ()


@pytest.mark.unit
def test_assignee_list_does_not_include_another_person() -> None:
    audit = _card("audit", "Initial audit template", "Dantino12")
    portal = _card("portal", "Sample portal card", "Fanuel Mengistu")
    response = answer_board_question("Tickets assigned to Fanual", (audit, portal))
    assert response is not None
    assert response.status == "answered"
    assert "Sample portal card" in response.answer
    assert "Initial audit template" not in response.answer


@pytest.mark.unit
def test_dante_matches_the_trello_username() -> None:
    audit = _card("audit", "Initial audit template", "Dantino12")
    response = answer_board_question("What tickets are assigned to Dante", (audit,))
    assert response is not None
    assert "Initial audit template" in response.answer
    assert "No indexed card" not in response.answer


@pytest.mark.unit
def test_members_line_uses_canonical_name_and_aliases() -> None:
    card = TrelloCard(
        card_id="card-1",
        name="Sample card",
        description="",
        list_name="Doing",
        labels=(),
        due=None,
        url="https://trello.com/c/abc",
        last_activity=datetime(2026, 11, 1, tzinfo=UTC),
        members=("Dantino12", "Fanuel Mengistu", "Anwar Hussein", "Sample Person"),
    )
    board = TrelloBoard("board", "Sample", "tIvVZqio", "https://trello.com/b/tIvVZqio/sample", (card,))
    text = units_for_card(board, card)[0].text
    assert "Dante Paniccia (aka Dante, Dantino, Dantino12)" in text
    assert "Fanuel Mengistu (aka Fanuel, Fanuel Anteneh, Fano)" in text
    assert "Anwar Hussein (aka Anwar)" in text
    assert "Sample Person" in text
