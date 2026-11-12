"""Mixed-source answers and remembered contributions. Fixtures are synthetic."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmark_cli import agent
from benchmark_cli.contributions import (
    THANKS_REPLY,
    UNKNOWN_INVITE,
    contribution_record,
    is_contribution_reply,
    prepare_contribution,
    remove_contribution,
    store_contribution,
)
from benchmark_cli.discord_ask import AddAnswerModal, format_discord_answer
from benchmark_cli.models import AnswerResponse, Citation
from benchmark_cli.retrieval import HybridSearchResult, RetrievalDecision, apply_source_quota
from benchmark_cli.storage.postgres import SearchChunk

_DIGEST = "ab" * 32


def _chunk(chunk_id: int, source_type: str, text: str, **metadata: object) -> SearchChunk:
    meta = {"source_type": source_type, **metadata}
    return SearchChunk(chunk_id, f"{source_type}/{chunk_id}.md", 0, text, meta, 1.0, content_sha256=_DIGEST)


def _noop_llm(**_kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=""))])


@pytest.mark.unit
def test_source_quota_keeps_a_knowledge_base_hit_beside_discord() -> None:
    discord_hits = [
        HybridSearchResult(_chunk(index, "discord", f"Discord note {index}", channel="general"), 1.0)
        for index in range(1, 8)
    ]
    kb = HybridSearchResult(_chunk(20, "kb_page", "Add a page from the portal.", title="Guide"), 0.2)
    meeting = HybridSearchResult(
        _chunk(21, "transcript", "We decided to keep the sample meeting note.", meeting="Standup"),
        0.1,
    )

    chosen = apply_source_quota([*discord_hits, kb, meeting], per_source=4, limit=12)
    kinds = {str(item.chunk.metadata["source_type"]) for item in chosen}

    assert "kb_page" in kinds
    assert "transcript" in kinds
    assert any(item.chunk.metadata["source_type"] == "discord" for item in chosen)


@pytest.mark.unit
def test_current_work_uses_doing_cards_not_an_old_discord_excerpt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.unit.test_live_question_regressions import _as_tuple, _row

    doing = (
        _row("Doing", "Sample portal sync", "d1", 1),
        _row("Doing", "Sample cli notes", "d2", 2),
        _row("Doing", "Sample kiosk timeline", "d3", 3),
        _row("Doing", "Sample board cleanup", "d4", 4),
    )
    discord = _chunk(
        9,
        "discord",
        "I've added a few new tickets to the Trello backlog for the sample project.",
        channel="general",
    )
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [_as_tuple(row) for row in doing])
    monkeypatch.setattr(agent, "call_llm", _noop_llm)
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("model unavailable")),
    )
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision(
            "accepted",
            (HybridSearchResult(discord, 0.4, vector_score=0.9, fts_score=1.0),),
            0.9,
            2,
        ),
    )

    response = agent.run_agent("What tickets are we currently working on?")

    assert "Sample portal sync" in response.answer
    assert "Sample cli notes" in response.answer
    assert "Sample kiosk timeline" in response.answer
    assert "Sample board cleanup" in response.answer
    assert "I've added a few new tickets" not in response.answer
    assert "Relevant retrieved excerpts" not in response.answer


@pytest.mark.unit
def test_varied_phrasings_keep_a_validated_model_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.unit.test_live_question_regressions import _as_tuple, _row

    doing = _row("Doing", "Sample portal sync", "d1", 1)
    kb = _chunk(30, "kb_page", "Open the section the page belongs in.", title="Adding a page")
    meeting = _chunk(31, "transcript", "The sample meeting decided to pause the pilot.", meeting="Planning")
    seen: dict[str, str] = {}

    def structured(**kwargs: object) -> AnswerResponse:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        blob = "\n".join(str(item.get("content", "")) for item in messages if isinstance(item, dict))
        seen["context"] = blob
        if "add a page" in blob.lower() or True:
            quote = "Open the section the page belongs in."
            source = kb
            answer = "Open the section the page belongs in."
        return AnswerResponse(
            status="answered",
            answer=answer,
            confidence=0.8,
            citations=[
                Citation(
                    source_file=source.source_file,
                    chunk_id=source.id,
                    content_sha256=_DIGEST,
                    quote=quote,
                )
            ],
        )

    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [_as_tuple(doing)])
    monkeypatch.setattr(agent, "call_llm", _noop_llm)
    monkeypatch.setattr(agent, "call_llm_structured", structured)
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision(
            "accepted",
            (
                HybridSearchResult(kb, 0.5, vector_score=0.8, fts_score=1.0),
                HybridSearchResult(meeting, 0.4, vector_score=0.8, fts_score=1.0),
            ),
            0.8,
            2,
        ),
    )

    page = agent.run_agent("how do I add a page")
    assert "Open the section the page belongs in." in page.answer
    assert "Relevant retrieved excerpts" not in page.answer
    assert "trello_board" in seen["context"]
    assert "Knowledge base" in seen["context"]
    assert "Meeting transcript" in seen["context"]

    def working(**kwargs: object) -> AnswerResponse:
        return AnswerResponse(
            status="answered",
            answer="The team is currently working on [Sample portal sync](https://trello.com/c/d1).",
            confidence=0.8,
            citations=[
                Citation(
                    source_file=doing.source_file,
                    chunk_id=doing.chunk_id,
                    content_sha256=doing.content_sha256,
                    quote="Card: Sample portal sync",
                )
            ],
        )

    monkeypatch.setattr(agent, "call_llm_structured", working)
    current = agent.run_agent("what's everyone working on")
    assert "Sample portal sync" in current.answer
    assert "https://trello.com/c/d1" in current.answer

    def invented(**_kwargs: object) -> AnswerResponse:
        return AnswerResponse(
            status="answered",
            answer="Doing has [Invented sample card](https://trello.com/c/nope).",
            confidence=0.9,
            citations=[
                Citation(
                    source_file=doing.source_file,
                    chunk_id=doing.chunk_id,
                    content_sha256=doing.content_sha256,
                    quote="Card: Sample portal sync",
                )
            ],
        )

    monkeypatch.setattr(agent, "call_llm_structured", invented)
    rejected = agent.run_agent("summarize the board")
    assert "Invented sample card" not in rejected.answer
    assert "https://trello.com/c/nope" not in rejected.answer


@pytest.mark.unit
def test_modal_submit_stores_redacted_metadata_and_a_later_answer_attributes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored: list[dict[str, object]] = []

    def embed(texts: list[str]) -> list[list[float]]:
        assert "sk-" not in texts[0]
        return [[0.1, 0.2]]

    def upsert(rows: list[dict[str, object]]) -> int:
        stored.extend(rows)
        return 1

    monkeypatch.setattr("benchmark_cli.contributions.embed_texts", embed)
    monkeypatch.setattr("benchmark_cli.contributions.upsert_chunks", upsert)

    secret = "sk-" + "a" * 20
    source = store_contribution(
        question="When is the sample standup?",
        text=f"It is at nine. token {secret}",
        author_id=42,
        author_name="Ada Example",
        channel_id=7,
        message_id=8,
        message_url="https://discord.com/channels/1/7/8",
        contributed_at="2026-11-09",
    )

    assert source.startswith("user_contributed/")
    row = stored[0]
    metadata = row["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["source_type"] == "user_contributed"
    assert metadata["question"] == "When is the sample standup?"
    assert metadata["author_id"] == "42"
    assert metadata["author_name"] == "Ada Example"
    assert metadata["contributed_at"] == "2026-11-09"
    assert metadata["url"] == "https://discord.com/channels/1/7/8"
    assert secret not in str(row["chunk_text"])
    assert "[REDACTED]" in str(row["chunk_text"])
    assert isinstance(row["embedding"], list)

    modal = AddAnswerModal(
        question="When is the sample standup?",
        author_id=42,
        author_name="Ada Example",
        channel_id=7,
        message_id=9,
        message_url="https://discord.com/channels/1/7/9",
    )
    modal.answer._value = "Standup is at nine."

    class _Response:
        def __init__(self) -> None:
            self.events: list[tuple[str, object]] = []

        async def defer(self, *, ephemeral: bool = False) -> None:
            self.events.append(("defer", ephemeral))

        async def send_message(self, content: str, *, ephemeral: bool = False) -> None:
            self.events.append(("send", content))

    class _Followup:
        def __init__(self, events: list[tuple[str, object]]) -> None:
            self.events = events

        async def send(self, content: str, *, ephemeral: bool = False) -> None:
            self.events.append(("followup", content))
            assert ephemeral is True

    class _User:
        id = 42
        display_name = "Ada Example"

    class _Interaction:
        user = _User()
        response = _Response()
        followup = _Followup(response.events)

    import asyncio

    interaction = _Interaction()
    asyncio.run(modal.on_submit(interaction))
    assert interaction.response.events[0] == ("defer", True)
    assert interaction.response.events[1] == ("followup", THANKS_REPLY)
    assert len(stored) == 2

    note = _chunk(
        50,
        "user_contributed",
        "According to Ada Example on 2026-11-09:\nStandup is at nine.",
        author_name="Ada Example",
        contributed_at="2026-11-09",
        question="When is the sample standup?",
    )
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [])
    monkeypatch.setattr(agent, "call_llm", _noop_llm)

    def attribute(**kwargs: object) -> AnswerResponse:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        blob = "\n".join(str(item.get("content", "")) for item in messages if isinstance(item, dict))
        assert "user_contributed" in blob
        assert "Ada Example" in blob
        return AnswerResponse(
            status="answered",
            answer="According to Ada Example on 2026-11-09, standup is at nine.",
            confidence=0.7,
            citations=[
                Citation(
                    source_file=note.source_file,
                    chunk_id=note.id,
                    content_sha256=_DIGEST,
                    quote="According to Ada Example on 2026-11-09:",
                )
            ],
        )

    monkeypatch.setattr(agent, "call_llm_structured", attribute)
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision(
            "accepted",
            (HybridSearchResult(note, 0.5, vector_score=0.8, fts_score=1.0),),
            0.8,
            1,
        ),
    )

    answer = agent.run_agent("When is the sample standup?")
    assert "According to Ada Example on 2026-11-09" in answer.answer
    assert "Relevant retrieved excerpts" not in answer.answer


@pytest.mark.unit
def test_secrets_are_redacted_and_bot_replies_are_not_contributions() -> None:
    secret = "sk-" + "b" * 20
    assert secret not in prepare_contribution(f"use {secret} please")
    assert "[REDACTED]" in prepare_contribution(f"use {secret} please")
    record = contribution_record(
        question="Where is the sample token?",
        text=f"It was {secret}",
        author_id=1,
        author_name="Ada Example",
        contributed_at="2026-11-09",
        channel_id=1,
        message_id=2,
        message_url="https://discord.com/channels/1/1/2",
    )
    assert secret not in str(record["chunk_text"])
    assert is_contribution_reply(
        author_is_bot=True,
        referenced_author_is_bot=True,
        referenced_text=UNKNOWN_INVITE,
    ) is False
    assert is_contribution_reply(
        author_is_bot=False,
        referenced_author_is_bot=False,
        referenced_text=UNKNOWN_INVITE,
    ) is False
    assert is_contribution_reply(
        author_is_bot=False,
        referenced_author_is_bot=True,
        referenced_text="The answer is in the guide.",
    ) is False
    assert is_contribution_reply(
        author_is_bot=False,
        referenced_author_is_bot=True,
        referenced_text=f"No relevant information.\n\n{UNKNOWN_INVITE}",
    ) is True


@pytest.mark.unit
def test_unknown_reply_invites_an_answer_and_remove_uses_the_source_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered = format_discord_answer(
        AnswerResponse(
            status="no_relevant_information",
            answer="No relevant information was found in the company document library.",
            confidence=0.0,
            citations=[],
        )
    )
    assert rendered.startswith("No relevant information")
    assert UNKNOWN_INVITE in rendered
    assert "Add answer" in rendered

    removed: list[str] = []
    monkeypatch.setattr(
        "benchmark_cli.contributions.delete_source",
        lambda source: removed.append(source),
    )
    remove_contribution("user_contributed/7/8.md")
    assert removed == ["user_contributed/7/8.md"]
    with pytest.raises(ValueError):
        remove_contribution("discord/general.md")
