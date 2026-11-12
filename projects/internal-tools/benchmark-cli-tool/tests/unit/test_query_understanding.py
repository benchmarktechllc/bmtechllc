"""Query understanding, roster rendering, and offline paraphrase replay."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark_cli.models import AnswerResponse
from benchmark_cli.query_understanding import QueryPlan, understand_query
from benchmark_cli.replay import check_expect, load_replay, replay_case, sample_board
from benchmark_cli.roster import render_members
from benchmark_cli.trello_answers import cards_for_person, person_card_payload, snapshot_payload
from benchmark_cli.trello_sync import TrelloBoard, TrelloCard, units_for_card

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "replay" / "paraphrases.json"
_DANTE = (
    "Dante Paniccia is assigned to [Ship the portal](https://trello.com/c/ship) on Doing "
    "and [Old portal import](https://trello.com/c/old) on Done."
)
_KIOSK = "Fanuel Mengistu is assigned to [Kiosk MVP](https://trello.com/c/kiosk) on Doing."


@pytest.mark.unit
def test_members_render_real_names_and_leave_unknown_names() -> None:
    rendered = render_members(["Dantino12", "Sample Person"])
    assert rendered.startswith("Dante Paniccia (Dante, Dantino, Dantino12)")
    assert rendered.endswith("Sample Person")
    board = TrelloBoard(
        "board-1",
        "BMTech General Board",
        "tIvVZqio",
        "https://trello.com/b/tIvVZqio/bmtech-general-board",
        (),
    )
    card = TrelloCard(
        "card-1",
        "Ship the portal",
        "",
        "Doing",
        (),
        None,
        "https://trello.com/c/ship",
        None,
        (),
        (),
        ("Dantino12",),
    )
    text = units_for_card(board, card)[0].text
    assert "Members: Dante Paniccia (Dante, Dantino, Dantino12)" in text


@pytest.mark.unit
def test_snapshot_roster_expands_raw_trello_usernames() -> None:
    payload = snapshot_payload(sample_board())
    roster = payload["roster"]
    assert isinstance(roster, list)
    names = {item["name"] for item in roster if isinstance(item, dict)}
    assert names == {"Dante Paniccia", "Fanuel Mengistu", "Anwar Hussein"}
    ship = next(card for card in payload["cards"] if isinstance(card, dict) and card["title"] == "Ship the portal")
    members = ship["members"]
    assert isinstance(members, list)
    assert members[0]["name"] == "Dante Paniccia"
    assert "Dantino12" in members[0]["trello"]
    ordered = [card.title for card in cards_for_person(sample_board(), "Dante")]
    assert ordered == ["Ship the portal", "Old portal import"]
    fanuel = [card["title"] for card in person_card_payload(sample_board(), "Fano")]
    assert fanuel == ["Kiosk MVP", "Retired kiosk notes"]


@pytest.mark.unit
def test_cheap_understanding_reads_the_roster_and_does_not_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake(**kwargs: object) -> QueryPlan:
        seen["tier"] = kwargs["tier"]
        seen["component"] = kwargs["component"]
        seen["messages"] = kwargs["messages"]
        return QueryPlan(searches=["tickets Dante Paniccia", "extra", "third"], person="Dante")

    monkeypatch.setattr("benchmark_cli.query_understanding.call_llm_structured", fake)

    plan = understand_query("Dante's tickets", "What tickets are assigned to Fanuel?")

    assert seen["tier"] == "cheap"
    assert seen["component"] == "query_understanding"
    messages = seen["messages"]
    assert isinstance(messages, list)
    blob = "\n".join(str(message["content"]) for message in messages)
    assert "do not answer" in blob.casefold()
    assert "Dante Paniccia" in blob
    assert "Fanuel Mengistu" in blob
    assert "Anwar Hussein" in blob
    assert "Dante's tickets" in blob
    assert "What tickets are assigned to Fanuel?" in blob
    assert plan.person == "Dante Paniccia"
    assert plan.searches == ["tickets Dante Paniccia", "extra"]


@pytest.mark.unit
def test_understanding_failure_uses_the_raw_question(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**_kwargs: object) -> QueryPlan:
        raise RuntimeError("cheap model unavailable")

    monkeypatch.setattr("benchmark_cli.query_understanding.call_llm_structured", boom)
    failed = understand_query("Tickets assigned to Fanual", "What tickets are assigned to Dante")
    assert failed.person is None
    assert failed.searches == ["Tickets assigned to Fanual"]

    def unknown(**_kwargs: object) -> QueryPlan:
        return QueryPlan(searches=["Fanuel Mengistu tickets"], person="Fanual")

    monkeypatch.setattr("benchmark_cli.query_understanding.call_llm_structured", unknown)
    typo = understand_query("Tickets assigned to Fanual")
    assert typo.person is None
    assert typo.searches == ["Fanuel Mengistu tickets"]

    def nickname(**_kwargs: object) -> QueryPlan:
        return QueryPlan(searches=["Fanuel Mengistu"], person="Fano")

    monkeypatch.setattr("benchmark_cli.query_understanding.call_llm_structured", nickname)
    assert understand_query("what is Fano doing").person == "Fanuel Mengistu"


@pytest.mark.unit
def test_paraphrase_replay_matches_the_recorded_plan(quiet: None) -> None:
    del quiet
    payload = load_replay(_FIXTURE)
    assert payload["sample_board"] is True
    ids = [case["id"] for case in payload["cases"]]
    assert len(ids) >= 12
    assert len(ids) == len(set(ids))
    for case in payload["cases"]:
        response, trace = replay_case(case, sample_board())
        assert check_expect(response, case) == []
        assert all(item in trace["queries"] for item in case["plan"]["searches"])
        assert case["question"].strip() in trace["queries"]
        if "how many employees" not in case["question"].casefold():
            assert trace["snapshot_present"] is True
        if case["plan"]["person"]:
            blob = json.dumps(trace["prompts"])
            assert case["plan"]["person"] in blob
            assert "roster" in blob


@pytest.fixture
def quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmark_cli import agent

    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)


@pytest.mark.unit
def test_abstention_is_after_the_model_sees_roster_names(monkeypatch: pytest.MonkeyPatch, quiet: None) -> None:
    from benchmark_cli import agent

    del quiet
    prompts: list[object] = []

    def structured(**kwargs: object) -> AnswerResponse:
        prompts.append(kwargs.get("messages"))
        return AnswerResponse(
            status="no_relevant_information",
            answer="Nothing in the pack names a matching card.",
            confidence=0.0,
            citations=[],
        )

    monkeypatch.setattr(agent, "call_llm_structured", structured)
    monkeypatch.setattr(
        agent,
        "call_llm",
        lambda **_kwargs: __import__("types").SimpleNamespace(
            choices=[__import__("types").SimpleNamespace(message=__import__("types").SimpleNamespace(tool_calls=None, content=""))]
        ),
    )
    monkeypatch.setattr(agent, "classify_query", lambda _question: "cheap")
    monkeypatch.setattr(agent, "trello_card_rows", lambda: list(sample_board()))
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: __import__(
            "benchmark_cli.retrieval", fromlist=["RetrievalDecision"]
        ).RetrievalDecision("no_relevant_information", (), 0.12, 0),
    )

    response = agent.run_agent("What is Fanuel currently working on")

    assert prompts
    blob = json.dumps(prompts[0])
    assert "Dante Paniccia" in blob
    assert "Fanuel Mengistu" in blob
    assert "Dantino12" in blob
    assert response.status == "no_relevant_information"
    assert "temporarily unavailable" not in response.answer.casefold()


@pytest.mark.unit
def test_person_filter_rejects_another_persons_card(monkeypatch: pytest.MonkeyPatch, quiet: None) -> None:
    from benchmark_cli import agent
    from benchmark_cli.query_understanding import QueryPlan

    del quiet
    prompts: list[object] = []

    def structured(**kwargs: object) -> AnswerResponse:
        prompts.append(kwargs.get("messages"))
        return AnswerResponse.model_validate(
            {
                "status": "answered",
                "answer": _KIOSK,
                "confidence": 0.8,
                "citations": [
                    {
                        "source_file": "trello/tIvVZqio/kiosk.md",
                        "chunk_id": 3,
                        "content_sha256": __import__("hashlib").sha256(b"kiosk").hexdigest(),
                        "quote": "Card: Kiosk MVP",
                    }
                ],
            }
        )

    monkeypatch.setattr(agent, "understand_query", lambda *_args, **_kwargs: QueryPlan(searches=["Dante Paniccia"], person="Dante Paniccia"))
    monkeypatch.setattr(agent, "call_llm_structured", structured)
    monkeypatch.setattr(
        agent,
        "call_llm",
        lambda **_kwargs: __import__("types").SimpleNamespace(
            choices=[__import__("types").SimpleNamespace(message=__import__("types").SimpleNamespace(tool_calls=None, content=""))]
        ),
    )
    monkeypatch.setattr(agent, "classify_query", lambda _question: "cheap")
    monkeypatch.setattr(agent, "trello_card_rows", lambda: list(sample_board()))
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: __import__(
            "benchmark_cli.retrieval", fromlist=["RetrievalDecision"]
        ).RetrievalDecision("no_relevant_information", (), 0.1, 0),
    )

    response = agent.run_agent("What tickets are assigned to Dante")

    evidence = json.loads(prompts[0][1]["content"].split("\n", 1)[1])
    assert evidence["person_filter"] == "Dante Paniccia"
    assert [card["title"] for card in evidence["person_cards"]] == ["Ship the portal", "Old portal import"]
    assert "Dante Paniccia" in json.dumps(evidence["trello_board"]["roster"])
    assert response.status == "answered"
    assert "Kiosk MVP" not in response.answer
    assert "Ship the portal" in response.answer
    assert "Old portal import" in response.answer

@pytest.mark.unit
def test_failed_model_does_not_list_someone_elses_doing_cards(monkeypatch: pytest.MonkeyPatch, quiet: None) -> None:
    from benchmark_cli import agent
    from benchmark_cli.query_understanding import QueryPlan

    del quiet
    monkeypatch.setattr(
        agent,
        "understand_query",
        lambda *_args, **_kwargs: QueryPlan(searches=["Fanuel Mengistu current work"], person="Fanuel Mengistu"),
    )
    monkeypatch.setattr(agent, "call_llm_structured", lambda **_kwargs: None)
    monkeypatch.setattr(
        agent,
        "call_llm",
        lambda **_kwargs: __import__("types").SimpleNamespace(
            choices=[
                __import__("types").SimpleNamespace(
                    message=__import__("types").SimpleNamespace(tool_calls=None, content="")
                )
            ]
        ),
    )
    monkeypatch.setattr(agent, "classify_query", lambda _question: "cheap")
    monkeypatch.setattr(agent, "trello_card_rows", lambda: list(sample_board()))
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: __import__(
            "benchmark_cli.retrieval", fromlist=["RetrievalDecision"]
        ).RetrievalDecision("no_relevant_information", (), 0.1, 0),
    )

    response = agent.run_agent("what is Fano doing right now")

    assert response.status == "answered"
    assert "Kiosk MVP" in response.answer
    assert "Retired kiosk notes" in response.answer
    assert "Completed:" in response.answer
    assert "Ship the portal" not in response.answer


@pytest.mark.unit
def test_recorded_dante_answer_is_kept_when_retrieval_was_rejected(quiet: None) -> None:
    del quiet
    case = next(item for item in load_replay(_FIXTURE)["cases"] if item["id"] == "dante-assigned")
    response, trace = replay_case(case, sample_board())
    assert response.status == "answered"
    assert response.answer == _DANTE
    assert trace["snapshot_present"] is True
    assert [citation.quote for citation in response.citations] == [
        "Card: Ship the portal",
        "Card: Old portal import",
    ]
