"""Typo-tolerant roster matching. No model calls and no database."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from benchmark_cli import agent
from benchmark_cli.models import AnswerResponse
from benchmark_cli.names import corrected_question, resolve_question
from benchmark_cli.retrieval import RetrievalDecision
from benchmark_cli.trello_answers import TrelloCardRow, answer_board_question, person_work_answer
from benchmark_cli.trello_sync import TrelloBoard, TrelloCard, units_for_card

_DIGEST = "ab" * 32


def _card(card_id: str, title: str, members: str, list_name: str = "Doing", chunk_id: int = 1) -> TrelloCardRow:
    return TrelloCardRow(
        "Sample Board",
        list_name,
        card_id,
        title,
        f"https://trello.com/c/{card_id}",
        chunk_id,
        f"trello/board/{card_id}.md",
        _DIGEST,
        f"Card: {title}\nList: {list_name}\nMembers: {members}\nURL: https://trello.com/c/{card_id}",
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


_UNCHANGED = (
    "Show me the Backlog tickets",
    "who's on the team?",
    "what's next after the kiosk MVP?",
    "is there anything about security on our plate?",
    "what's the deal with the used car project?",
    "How many employees does BMTech have?",
    "Which channels are synced?",
    "what does Sev 2 mean?",
    "What's on the board?",
)
_TITLE_WORDS = (
    "Backlog",
    "security",
    "team",
    "kiosk",
    "used",
    "car",
    "Trello",
    "portal",
    "Assessment",
    "Playbook",
)


@pytest.mark.unit
@pytest.mark.parametrize("question", _UNCHANGED)
def test_exact_vocabulary_does_not_change_pipeline_inputs(question: str) -> None:
    resolution = resolve_question(question, _TITLE_WORDS)
    prompt = f"{resolution.note}\n\n{question}" if resolution.note else question
    assert resolution.changed is False
    assert resolution.note == ""
    assert resolution.query == " ".join(question.split())
    assert corrected_question(question, resolution) == question
    assert prompt == question


@pytest.mark.unit
def test_agent_passes_those_questions_through_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def retrieve(question_text: str, **_kwargs: object) -> RetrievalDecision:
        seen["retrieval"] = question_text
        return RetrievalDecision("no_relevant_information", (), None, 0)

    def complete(**kwargs: object) -> SimpleNamespace:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                seen["prompt"] = str(message["content"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=""))])

    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [])
    monkeypatch.setattr(agent, "retrieve_evidence", retrieve)
    monkeypatch.setattr(agent, "call_llm", complete)
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: AnswerResponse(
            status="no_relevant_information",
            answer="No relevant information was found in the company document library.",
            confidence=0.0,
        ),
    )

    for question in _UNCHANGED:
        seen.clear()
        agent.run_agent(question)
        assert seen["retrieval"] == question
        if "prompt" in seen:
            assert seen["prompt"] == question


def _roster() -> tuple[TrelloCardRow, ...]:
    dante = (
        ("playbook", "Assessment Playbook", "Doing", 1),
        ("audit", "Initial audit template", "Sev 1", 2),
        ("done-a", "Sample done one", "Done", 3),
        ("done-b", "Sample done two", "Done", 4),
        ("done-c", "Sample done three", "Done", 5),
        ("done-d", "Sample done four", "Done", 6),
    )
    fanuel = (
        ("do-1", "Sample fanuel doing one", "Doing", 7),
        ("do-2", "Sample fanuel doing two", "Doing", 8),
        ("ready", "Client readiness assessment of the RAG system.", "Done", 9),
        ("model", "Model performance evaluation standards.", "Done", 10),
        ("ethiopia", "Research regarding business setup in Ethiopia", "Sev 3", 11),
    )
    rows = [_card(card_id, title, "Dantino12", list_name, chunk_id) for card_id, title, list_name, chunk_id in dante]
    rows.extend(
        _card(card_id, title, "Fanuel Mengistu", list_name, chunk_id)
        for card_id, title, list_name, chunk_id in fanuel
    )
    rows.append(_card("anwar-do", "Sample anwar card", "Anwar Hussein", "Doing", 12))
    rows.append(_card("anwar-done", "Sample anwar done card", "Anwar Hussein", "Done", 13))
    return tuple(rows)


@pytest.mark.unit
def test_dante_and_fanuel_include_every_list_with_done_last() -> None:
    rows = _roster()
    dante = answer_board_question("What tickets are assigned to Dante", rows)
    fanuel = answer_board_question("Tickets assigned to Fanual", rows)
    assert dante is not None and fanuel is not None
    for title in (
        "Assessment Playbook",
        "Initial audit template",
        "Sample done one",
        "Sample done two",
        "Sample done three",
        "Sample done four",
    ):
        assert title in dante.answer
    assert "Sample fanuel doing one" not in dante.answer
    assert dante.answer.index("Trello cards in Doing") < dante.answer.index("Trello cards in Done")
    for title in (
        "Sample fanuel doing one",
        "Sample fanuel doing two",
        "Client readiness assessment of the RAG system.",
        "Model performance evaluation standards.",
        "Research regarding business setup in Ethiopia",
    ):
        assert title in fanuel.answer
    assert "Initial audit template" not in fanuel.answer
    assert "Trello cards in Sev 3" in fanuel.answer
    assert fanuel.answer.index("Trello cards in Doing") < fanuel.answer.index("Trello cards in Done")


@pytest.mark.unit
@pytest.mark.parametrize(
    "question",
    ("what is anwr working on", "what's dantino working on"),
)
def test_working_on_uses_the_person_list(question: str) -> None:
    rows = _roster()
    response = person_work_answer(question, rows)
    assert response is not None and response.status == "answered"
    if "anwr" in question:
        assert "Sample anwar card" in response.answer
        assert "Sample anwar done card" in response.answer
        assert response.answer.index("Sample anwar card") < response.answer.index("Sample anwar done card")
        assert "Assessment Playbook" not in response.answer
    else:
        assert "Assessment Playbook" in response.answer
        assert "Sample done one" in response.answer
        assert response.answer.index("Assessment Playbook") < response.answer.index("Sample done one")
        assert "Sample fanuel doing one" not in response.answer


@pytest.mark.unit
def test_trelo_corrects_without_a_snapshot_and_reaches_the_board() -> None:
    resolution = resolve_question("what's on the trelo board")
    corrected = corrected_question("what's on the trelo board", resolution)
    assert resolution.changed
    assert "Trello" in resolution.query
    assert "what's on the Trello board" == corrected.casefold() or "trello" in corrected.casefold()
    board = answer_board_question(corrected, (_card("one", "Sample backlog card", "", "Backlog", 1),))
    assert board is not None and "Backlog" in board.answer


@pytest.mark.unit
def test_two_edits_match_only_longer_tokens() -> None:
    long_typo = resolve_question("what is fireflxys")
    assert any(hit.canonical == "Fireflies" for hit in long_typo.hits)
    short_typo = resolve_question("what is trelo")
    assert any(hit.canonical == "Trello" for hit in short_typo.hits)
    too_far = resolve_question("what is fireflxyz")
    assert too_far.changed is False
