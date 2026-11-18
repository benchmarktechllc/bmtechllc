"""Paraphrases that must be answered from the whole indexed board."""

from __future__ import annotations

import pytest

from benchmark_cli import agent
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
    from types import SimpleNamespace

    from benchmark_cli.retrieval import RetrievalDecision

    monkeypatch.setattr(agent, "trello_card_rows", lambda: [_as_tuple(row) for row in rows])
    monkeypatch.setattr(
        agent,
        "call_llm",
        lambda **_kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=""))]
        ),
    )

    def _no_draft(**_kwargs: object) -> None:
        raise RuntimeError("snapshot unavailable")

    monkeypatch.setattr(agent, "call_llm_structured", _no_draft)
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision("no_relevant_information", (), None, 0),
    )
    return rows


@pytest.fixture
def quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)


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


def _silence(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from benchmark_cli.retrieval import RetrievalDecision

    monkeypatch.setattr(
        agent,
        "call_llm",
        lambda **_kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=""))]
        ),
    )
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision("no_relevant_information", (), None, 0),
    )

    def _no_draft(**_kwargs: object) -> None:
        raise RuntimeError("snapshot unavailable")

    monkeypatch.setattr(agent, "call_llm_structured", _no_draft)


@pytest.mark.unit
def test_snapshot_is_sent_to_the_model_and_a_valid_draft_is_kept(
    monkeypatch: pytest.MonkeyPatch, quiet: None
) -> None:
    import json

    _board(monkeypatch)
    seen: dict[str, object] = {}

    def structured(**kwargs: object) -> AnswerResponse:
        seen["messages"] = kwargs["messages"]
        return AnswerResponse(
            status="answered",
            answer=(
                "from the indexed snapshot\n"
                "Backlog 11, Doing 4, Done 12, Sev 1 12, Sev 2 2, Sev 3 3\n"
                "- [Sample MCP project](https://trello.com/c/c1)"
            ),
            confidence=0.9,
            citations=[
                Citation(
                    source_file="trello/tIvVZqio/c1.md",
                    chunk_id=1,
                    content_sha256="a" * 64,
                    quote="Card: Sample MCP project",
                )
            ],
        )

    monkeypatch.setattr(agent, "call_llm_structured", structured)

    response = agent.run_agent("What's on the Trello board?")

    messages = seen["messages"]
    assert isinstance(messages, list)
    raw = str(messages[1]["content"])
    payload = json.loads(raw[raw.index("{") :])
    titles = {card["title"] for card in payload["trello_board"]["cards"]}
    assert len(titles) == 44
    assert payload["trello_board"]["counts"]["Backlog"] == 11
    assert payload["trello_board"]["counts"]["Done"] == 12
    assert "from the indexed snapshot" in response.answer
    assert "Sample MCP project" in response.answer

