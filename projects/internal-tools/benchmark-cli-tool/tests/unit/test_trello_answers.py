"""List-membership answers from indexed Trello cards, without retrieval."""

from __future__ import annotations

import hashlib

import pytest

from benchmark_cli import agent
from benchmark_cli.discord_ask import format_discord_answer
from benchmark_cli.retrieval import RetrievalDecision
from benchmark_cli.trello_answers import (
    MAX_LISTED_CARDS,
    TrelloCardRow,
    requested_trello_list,
    trello_list_answer,
)

_TOKEN = "ghp_" + ("a" * 36)
_MEMBERSHIP = (
    "Which tickets are in backlog?",
    "Which Trello cards are in Doing?",
    "which tickets are in active in progress state?",
    'Which cards are in the "Done" column?',
    "Which tickets are under Review?",
)
_NOT_MEMBERSHIP = (
    "How does the Trello integration work?",
    "What is Trello?",
    "Which Trello board do we sync?",
    "Why is this Trello card missing?",
    "How do I configure the Trello API token?",
    "What tickets did Dante add to the backlog for RAG?",
    "Which Discord channels are synced?",
    "What is the link to the knowledge base portal?",
    "In doing state?",
    "How do I move a ticket into the in progress column?",
)


def _digest(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def _row(
    list_name: str,
    title: str,
    card_id: str,
    chunk_id: int,
    *,
    description: str = "",
    digest: str | None = None,
    url: str | None = None,
    board: str = "BMTech General Board",
) -> TrelloCardRow:
    card_url = url if url is not None else f"https://trello.com/c/{card_id}"
    chunk = f"Card: {title}\nBoard: {board}\nList: {list_name}\nURL: {card_url}"
    if description:
        chunk = f"{chunk}\n\n{description}"
    return TrelloCardRow(
        board=board,
        list_name=list_name,
        card_id=card_id,
        title=title,
        url=card_url,
        chunk_id=chunk_id,
        source_file=f"trello/tIvVZqio/{card_id}.md",
        content_sha256=digest if digest is not None else _digest(card_id),
        chunk_text=chunk,
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


_ROWS = (
    _row("Backlog", "Nightly Discord sync", "aaa", 1),
    _row(
        "Doing",
        "Notion data job",
        "bbb",
        2,
        description=f"Secret desc {_TOKEN}",
    ),
    _row("Done", "Portal ingestion", "ccc", 3),
)


@pytest.fixture
def quiet_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "call_llm", lambda **_kwargs: pytest.fail("model should not run"))
    monkeypatch.setattr(
        agent, "call_llm_structured", lambda **_kwargs: pytest.fail("model should not run")
    )
    monkeypatch.setattr(
        agent, "retrieve_evidence", lambda *_args, **_kwargs: pytest.fail("retrieval should not run")
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("question", "expected"),
    (
        ("Which tickets are in backlog?", "backlog"),
        ("Which Trello cards are in Doing?", "doing"),
        ("which tickets are in active in progress state?", "doing"),
        ('Which cards are in the "Done" column?', "done"),
        ("Which tickets are under Review?", "review"),
        ("which tickets are in in-progress?", "doing"),
        ("list the trello cards in the Doing list", "doing"),
    ),
)
def test_membership_questions_name_one_list(question: str, expected: str) -> None:
    assert requested_trello_list(question) == expected


@pytest.mark.unit
@pytest.mark.parametrize("question", _NOT_MEMBERSHIP)
def test_other_questions_are_not_list_lookups(question: str) -> None:
    assert question not in _MEMBERSHIP
    assert requested_trello_list(question) is None


@pytest.mark.unit
def test_active_in_progress_lists_only_the_doing_card(
    monkeypatch: pytest.MonkeyPatch, quiet_agent: None
) -> None:
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [_as_tuple(row) for row in _ROWS])

    response = agent.run_agent("which tickets are in active in progress state?")

    assert response.status == "answered"
    assert response.confidence == 0.8
    assert "Notion data job" in response.answer
    assert "https://trello.com/c/bbb" in response.answer
    assert "Nightly Discord sync" not in response.answer
    assert "Portal ingestion" not in response.answer
    assert [citation.quote for citation in response.citations] == ["Card: Notion data job"]
    rendered = response.model_dump_json()
    assert _TOKEN not in rendered
    assert "Secret desc" not in rendered


@pytest.mark.unit
@pytest.mark.parametrize(
    ("question", "title", "absent"),
    (
        ("Which tickets are in backlog?", "Nightly Discord sync", "Notion data job"),
        ("Which Trello cards are in Doing?", "Notion data job", "Portal ingestion"),
        ('Which cards are in the "Done" column?', "Portal ingestion", "Nightly Discord sync"),
    ),
)
def test_named_lists_return_only_matching_titles(
    monkeypatch: pytest.MonkeyPatch,
    quiet_agent: None,
    question: str,
    title: str,
    absent: str,
) -> None:
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [_as_tuple(row) for row in _ROWS])

    response = agent.run_agent(question)

    assert response.status == "answered"
    assert title in response.answer
    assert absent not in response.answer
    assert response.citations[0].quote == f"Card: {title}"


@pytest.mark.unit
def test_under_review_uses_the_named_list(
    monkeypatch: pytest.MonkeyPatch, quiet_agent: None
) -> None:
    review = _row("Review", "Portal copy", "rev", 4)
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [_as_tuple(review)])

    response = agent.run_agent("Which tickets are under Review?")

    assert response.status == "answered"
    assert "Portal copy" in response.answer
    assert response.citations[0].quote == "Card: Portal copy"


@pytest.mark.unit
def test_unknown_list_is_explicit_and_does_not_list_cards() -> None:
    response = trello_list_answer("qa", _ROWS)

    assert response.status == "no_relevant_information"
    assert response.confidence == 0.0
    assert 'named "qa"' in response.answer
    assert "Backlog, Doing, Done" in response.answer
    for row in _ROWS:
        assert row.title not in response.answer
    assert response.citations == []


@pytest.mark.unit
def test_empty_index_says_no_cards_are_indexed(
    monkeypatch: pytest.MonkeyPatch, quiet_agent: None
) -> None:
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [])

    response = agent.run_agent("which trello tickets are in backlog")

    assert response.status == "no_relevant_information"
    assert "No Trello cards are in the retrieval index yet" in response.answer
    assert response.citations == []


@pytest.mark.unit
def test_duplicate_cards_are_listed_once_and_long_lists_are_capped() -> None:
    rows = [_row("Backlog", f"Card {i:02d}", f"id{i}", i + 1) for i in range(MAX_LISTED_CARDS + 5)]
    rows.append(rows[0])

    response = trello_list_answer("backlog", rows)

    assert response.answer.count("Card 00") == 1
    assert len(response.citations) == MAX_LISTED_CARDS
    assert "…and 5 more." in response.answer
    assert len(format_discord_answer(response)) <= 2000


@pytest.mark.unit
def test_secret_in_a_title_is_redacted() -> None:
    response = trello_list_answer("backlog", [_row("Backlog", f"Rotate {_TOKEN}", "z9", 9)])

    assert _TOKEN not in response.answer
    assert all(_TOKEN not in citation.quote for citation in response.citations)
    assert "[REDACTED]" in response.answer


@pytest.mark.unit
def test_cards_without_a_hash_are_not_cited() -> None:
    blank = _row("Backlog", "Missing hash", "nohash", 8, digest="")
    kept = _row("Backlog", "Kept card", "kept", 9)

    response = trello_list_answer("backlog", [blank, kept])

    assert "Missing hash" not in response.answer
    assert "Kept card" in response.answer
    assert len(response.citations) == 1


@pytest.mark.unit
def test_phrase_boundary_matches_a_suffix_and_not_a_longer_word() -> None:
    review = trello_list_answer("review", [_row("Code Review", "Style pass", "style", 5)])
    reviewing = trello_list_answer("review", [_row("Reviewing", "Still open", "open", 6)])

    assert "Style pass" in review.answer
    assert reviewing.status == "no_relevant_information"
    assert "Still open" not in reviewing.answer


@pytest.mark.unit
def test_in_progress_list_matches_a_doing_request() -> None:
    response = trello_list_answer("doing", [_row("In Progress", "Queue drain", "queue", 7)])

    assert "Queue drain" in response.answer


@pytest.mark.unit
def test_non_https_url_is_not_linked() -> None:
    response = trello_list_answer(
        "backlog",
        [_row("Backlog", "Internal", "local", 11, url="http://trello.example/c/local")],
    )

    assert response.answer.endswith("- Internal") or "\n- Internal" in response.answer
    assert "http://trello.example" not in response.answer


@pytest.mark.unit
@pytest.mark.parametrize("question", _NOT_MEMBERSHIP)
def test_non_membership_questions_fall_through_to_retrieval(
    monkeypatch: pytest.MonkeyPatch, quiet_agent: None, question: str
) -> None:
    monkeypatch.setattr(agent, "trello_card_rows", lambda: pytest.fail("list lookup"))
    seen: dict[str, str] = {}

    def retrieve(question_text: str, **_kwargs: object) -> RetrievalDecision:
        seen["question"] = question_text
        return RetrievalDecision("no_relevant_information", (), None, 0)

    monkeypatch.setattr(agent, "retrieve_evidence", retrieve)

    response = agent.run_agent(question)

    assert seen["question"] == question
    assert response.status == "no_relevant_information"


@pytest.mark.unit
def test_credential_question_does_not_list_cards(
    monkeypatch: pytest.MonkeyPatch, quiet_agent: None
) -> None:
    monkeypatch.setattr(agent, "trello_card_rows", lambda: pytest.fail("list lookup"))

    response = agent.run_agent("which tickets are in Doing and what is the api key?")

    assert "api key" not in response.answer.casefold()
    assert "credentials" in response.answer


@pytest.mark.unit
def test_unavailable_index_falls_through_to_retrieval(
    monkeypatch: pytest.MonkeyPatch, quiet_agent: None
) -> None:
    def unavailable() -> list[object]:
        raise RuntimeError("database down")

    monkeypatch.setattr(agent, "trello_card_rows", unavailable)
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision("no_relevant_information", (), None, 0),
    )

    response = agent.run_agent("which trello tickets are in backlog")

    assert response.status == "no_relevant_information"
    assert "No Trello cards are in the retrieval index yet" not in response.answer
