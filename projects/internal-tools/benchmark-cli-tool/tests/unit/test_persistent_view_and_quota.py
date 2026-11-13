"""The Add answer button still works after a restart, and a 429 does not look like 'unknown'."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from benchmark_cli import agent
from benchmark_cli.agent import MODEL_UNAVAILABLE_REPLY
from benchmark_cli.discord_ask import (
    ADD_ANSWER_CUSTOM_ID,
    AddAnswerModal,
    AddAnswerView,
    AskClient,
    AskConfig,
)
from benchmark_cli.llm import _quota_error
from benchmark_cli.retrieval import HybridSearchResult, RetrievalDecision
from benchmark_cli.storage.postgres import SearchChunk


class _RateLimit(Exception):
    status_code = 429


class _Channel:
    def __init__(self, original: SimpleNamespace) -> None:
        self.id = 55
        self.original = original

    async def fetch_message(self, message_id: int) -> SimpleNamespace:
        assert message_id == self.original.id
        return self.original


def _question_message() -> tuple[SimpleNamespace, SimpleNamespace]:
    channel = _Channel(SimpleNamespace())
    original = channel.original
    original.id = 42
    original.content = "When is the sample standup?"
    original.channel = channel
    original.guild = SimpleNamespace(id=9)
    bot_message = SimpleNamespace(
        id=99,
        content="I couldn't find this.",
        reference=SimpleNamespace(resolved=None, message_id=original.id),
        channel=channel,
        guild=SimpleNamespace(id=9),
    )
    return original, bot_message


@pytest.mark.unit
def test_add_answer_click_after_restart_opens_the_modal_from_the_referenced_message() -> None:
    sent_before_restart = AddAnswerView()
    assert sent_before_restart.timeout is None
    assert sent_before_restart.is_persistent()
    assert sent_before_restart.children[0].custom_id == ADD_ANSWER_CUSTOM_ID
    del sent_before_restart

    client = AskClient(AskConfig(100000000000000001, None))
    registered: list[AddAnswerView] = []

    def add_view(view: AddAnswerView, *, message_id: int | None = None) -> None:
        assert message_id is None
        assert view.is_persistent()
        registered.append(view)

    async def sync() -> None:
        return None

    client.add_view = add_view  # type: ignore[method-assign]
    client.sync_commands = sync  # type: ignore[method-assign]
    asyncio.run(client.setup_hook())

    view = registered[0]
    assert not hasattr(view, "question")
    _original, bot_message = _question_message()
    opened: dict[str, AddAnswerModal] = {}

    class _Response:
        async def send_modal(self, modal: AddAnswerModal) -> None:
            opened["modal"] = modal

    interaction = SimpleNamespace(
        message=bot_message,
        user=SimpleNamespace(id=7, display_name="Ada Example", name="Ada"),
        response=_Response(),
    )
    asyncio.run(view.children[0].callback(interaction))

    modal = opened["modal"]
    assert modal.question == "When is the sample standup?"
    assert modal.author_name == "Ada Example"
    assert modal.message_id == 42
    assert "42" in modal.message_url


@pytest.mark.unit
def test_rate_limit_says_the_service_is_unavailable_but_trello_lists_still_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.unit.test_live_question_regressions import _as_tuple, _row

    assert _quota_error(_RateLimit("slow down")) is not None
    doing = _row("Doing", "Sample portal sync", "d1", 1)
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [_as_tuple(doing)])
    monkeypatch.setattr(agent, "call_llm", lambda **_kwargs: (_ for _ in ()).throw(_RateLimit("429")))
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: (_ for _ in ()).throw(_RateLimit("429")),
    )
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision("no_relevant_information", (), None, 0),
    )

    listed = agent.run_agent("Which Trello cards are in Doing?")

    assert "Sample portal sync" in listed.answer
    assert "temporarily unavailable" not in listed.answer.casefold()

    page = SearchChunk(
        8,
        "portal/guide/adding-a-page.md",
        0,
        "Open the section the page belongs in.",
        {"source_type": "kb_page", "title": "Adding a page"},
        1.0,
        content_sha256="ab" * 32,
    )
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [])
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision(
            "accepted",
            (HybridSearchResult(page, 0.4, vector_score=0.8, fts_score=1.0),),
            0.8,
            1,
        ),
    )

    unavailable = agent.run_agent("How do I add a page to the KB?")

    assert unavailable.answer == MODEL_UNAVAILABLE_REPLY
    assert unavailable.citations == []
