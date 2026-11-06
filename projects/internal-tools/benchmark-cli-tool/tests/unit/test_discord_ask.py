"""Offline tests for Discord reply formatting and ask-channel gating."""

from __future__ import annotations

import discord
import pytest

from benchmark_cli.discord_ask import (
    DISCORD_MESSAGE_LIMIT,
    THREAD_NAME_LIMIT,
    ask_command,
    format_discord_answer,
    gateway_intents,
    in_ask_channel,
    load_runtime,
    parse_snowflake,
    should_answer_channel_message,
    slash_reply_is_ephemeral,
    thread_name_for,
    truncate_discord,
)
from benchmark_cli.models import AnswerResponse, Citation

ASK_CHANNEL_ID = 100000000000000001
OTHER_CHANNEL_ID = 100000000000000002
BOT_USER_ID = 100000000000000010
USER_ID = 100000000000000011
SNOWFLAKE = "123456789012345678"


def _citation(quote: str, *, chunk_id: int = 7, source: str = "policies/remote.txt") -> Citation:
    return Citation(
        source_file=source,
        chunk_id=chunk_id,
        content_sha256="ab" * 32,
        quote=quote,
    )


def _answered(answer: str, *citations: Citation) -> AnswerResponse:
    return AnswerResponse(
        status="answered",
        answer=answer,
        confidence=0.8,
        citations=list(citations) or [_citation("Remote work is allowed.")],
    )


@pytest.mark.unit
def test_short_answer_includes_citation_quote_and_source() -> None:
    rendered = format_discord_answer(
        _answered("Use the home-office stipend.", _citation("Employees may expense a desk."))
    )

    assert "Use the home-office stipend." in rendered
    assert "policies/remote.txt" in rendered
    assert "chunk 7" in rendered
    assert "Employees may expense a desk." in rendered
    assert "**Citations**" in rendered
    assert len(rendered) <= DISCORD_MESSAGE_LIMIT


@pytest.mark.unit
def test_abstention_has_no_citation_block() -> None:
    rendered = format_discord_answer(
        AnswerResponse(
            status="no_relevant_information",
            answer="No relevant information was found in the company document library.",
            confidence=0.0,
            citations=[],
        )
    )

    assert rendered.startswith("No relevant information")
    assert "Citations" not in rendered


@pytest.mark.unit
def test_long_answer_keeps_citations_inside_the_discord_limit() -> None:
    rendered = format_discord_answer(
        _answered("A" * 5000, _citation("B" * 800, chunk_id=3, source="handbook.md")),
        limit=500,
    )

    assert len(rendered) <= 500
    assert "…(truncated)" in rendered
    assert "**Citations**" in rendered
    assert "handbook.md" in rendered
    assert "chunk 3" in rendered


@pytest.mark.unit
def test_oversized_citation_block_is_still_capped() -> None:
    citations = [
        _citation("quote " * 40, chunk_id=index, source=f"source-{index}.txt")
        for index in range(1, 12)
    ]
    rendered = format_discord_answer(_answered("Answer.", *citations), limit=350)

    assert len(rendered) <= 350
    assert rendered.endswith("…(truncated)")


@pytest.mark.unit
def test_mentions_in_answers_are_neutralized() -> None:
    rendered = format_discord_answer(
        _answered(
            "Ping @everyone and <@123456789012345678> about @here.",
            _citation("See the on-call rota."),
        )
    )

    assert "@everyone" not in rendered.replace("@\u200beveryone", "")
    assert "@\u200beveryone" in rendered
    assert "@\u200bhere" in rendered
    assert "@\u200b123456789012345678" in rendered


@pytest.mark.unit
def test_truncate_discord_keeps_short_text_and_caps_long_text() -> None:
    assert truncate_discord("hello", limit=5) == "hello"
    capped = truncate_discord("hello there friend", limit=16)
    assert capped.endswith("…(truncated)")
    assert len(capped) == 16
    assert len(truncate_discord("x" * 50, limit=5)) == 5


@pytest.mark.unit
def test_truncate_discord_rejects_a_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        truncate_discord("hello", limit=0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("channel_id", "parent_id", "expected"),
    [
        (ASK_CHANNEL_ID, None, True),
        (999, ASK_CHANNEL_ID, True),
        (OTHER_CHANNEL_ID, None, False),
        (888, OTHER_CHANNEL_ID, False),
    ],
)
def test_in_ask_channel_includes_child_threads(channel_id: int, parent_id: int | None, expected: bool) -> None:
    assert (
        in_ask_channel(channel_id=channel_id, parent_channel_id=parent_id, ask_channel_id=ASK_CHANNEL_ID)
        is expected
    )


def _message(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "channel_id": ASK_CHANNEL_ID,
        "parent_channel_id": None,
        "author_is_bot": False,
        "author_id": USER_ID,
        "bot_user_id": BOT_USER_ID,
        "content": "What is the PTO policy?",
        "message_type": discord.MessageType.default,
        "ask_channel_id": ASK_CHANNEL_ID,
    }
    payload.update(overrides)
    return payload


@pytest.mark.unit
def test_channel_listener_answers_human_text_in_the_ask_channel_and_its_threads() -> None:
    assert should_answer_channel_message(**_message()) is True
    assert should_answer_channel_message(**_message(message_type=discord.MessageType.reply)) is True
    assert (
        should_answer_channel_message(
            **_message(channel_id=555, parent_channel_id=ASK_CHANNEL_ID)
        )
        is True
    )


@pytest.mark.unit
def test_channel_listener_ignores_bots_other_channels_and_non_text() -> None:
    assert should_answer_channel_message(**_message(author_is_bot=True)) is False
    assert should_answer_channel_message(**_message(author_id=BOT_USER_ID)) is False
    assert should_answer_channel_message(**_message(content="   ")) is False
    assert should_answer_channel_message(**_message(channel_id=OTHER_CHANNEL_ID)) is False
    assert (
        should_answer_channel_message(
            **_message(channel_id=555, parent_channel_id=OTHER_CHANNEL_ID)
        )
        is False
    )
    assert (
        should_answer_channel_message(**_message(message_type=discord.MessageType.chat_input_command))
        is False
    )
    assert (
        should_answer_channel_message(**_message(message_type=discord.MessageType.thread_created))
        is False
    )


@pytest.mark.unit
def test_gateway_requests_the_message_content_intent() -> None:
    intents = gateway_intents()
    assert intents.message_content is True
    assert intents.guilds is True


@pytest.mark.unit
def test_slash_command_is_available_guild_wide_with_a_required_question() -> None:
    parameters = list(ask_command.parameters)
    assert ask_command.name == "ask"
    assert ask_command.guild_only is True
    assert [(parameter.name, parameter.required) for parameter in parameters] == [("question", True)]
    assert slash_reply_is_ephemeral(question="   ") is True
    assert slash_reply_is_ephemeral(question="Where is the runbook?") is False


@pytest.mark.unit
def test_thread_name_collapses_whitespace_and_stays_within_discord_limit() -> None:
    assert thread_name_for("  quarterly\nplan  ") == "quarterly plan"
    assert thread_name_for("   ") == "Question"
    long_name = thread_name_for("word " * 50)
    assert len(long_name) <= THREAD_NAME_LIMIT
    assert long_name.endswith("…")


@pytest.mark.unit
def test_parse_snowflake_accepts_channel_ids_only() -> None:
    assert parse_snowflake(f"  {SNOWFLAKE}\n", name="DISCORD_ASK_CHANNEL_ID") == int(SNOWFLAKE)
    with pytest.raises(ValueError, match="DISCORD_ASK_CHANNEL_ID"):
        parse_snowflake("ask-bmtech", name="DISCORD_ASK_CHANNEL_ID")
    with pytest.raises(ValueError, match="DISCORD_ASK_GUILD_ID"):
        parse_snowflake("12345", name="DISCORD_ASK_GUILD_ID")


@pytest.mark.unit
def test_load_runtime_reports_missing_names_without_values(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DISCORD_ANSWER_BOT_TOKEN",
        "DISCORD_ASK_CHANNEL_ID",
        "DATABASE_URL",
        "OPENAI_API_KEY",
        "DISCORD_ASK_GUILD_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "your_openai_api_key_here")

    with pytest.raises(SystemExit, match="DISCORD_ANSWER_BOT_TOKEN") as exc_info:
        load_runtime()

    message = str(exc_info.value)
    assert "OPENAI_API_KEY" in message
    assert "your_openai_api_key_here" not in message


@pytest.mark.unit
def test_load_runtime_reads_optional_guild_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISCORD_ANSWER_BOT_TOKEN", "token-value")
    monkeypatch.setenv("DISCORD_ASK_CHANNEL_ID", SNOWFLAKE)
    monkeypatch.setenv("DATABASE_URL", "postgresql://benchmark@localhost/benchmark")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("DISCORD_ASK_GUILD_ID", "223456789012345678")

    token, config = load_runtime()

    assert token == "token-value"
    assert config.ask_channel_id == int(SNOWFLAKE)
    assert config.guild_id == 223456789012345678
