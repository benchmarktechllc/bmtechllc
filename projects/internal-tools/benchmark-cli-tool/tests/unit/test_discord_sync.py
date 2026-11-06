"""Offline contract tests for Discord's scheduled ingestion connector."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from benchmark_cli import discord_sync
from benchmark_cli.ingestion.clean import discord_units_from_messages
from benchmark_cli.storage.postgres import DiscordSourceMessage


@pytest.mark.unit
def test_live_discord_messages_use_the_same_redaction_and_bot_filtering() -> None:
    units = discord_units_from_messages(
        [
            {
                "id": "1",
                "type": 0,
                "timestamp": "2026-10-21T01:02:03Z",
                "author": {"global_name": None, "username": "Ada"},
                "content": "password=very-secret <@42>",
            },
            {
                "id": "2",
                "type": 0,
                "timestamp": "2026-10-21T01:03:03Z",
                "author": {"username": "Deploy Bot", "bot": True},
                "content": "ignore me",
            },
        ],
        channel="engineering",
        users={"42": "Grace"},
    )

    assert len(units) == 1
    assert units[0].text == "[REDACTED] @Grace"
    assert units[0].extra["channel"] == "engineering"
    assert units[0].extra["message_id"] == "1"


@pytest.mark.unit
def test_message_fetches_until_it_includes_the_saved_cursor() -> None:
    class FakeClient(discord_sync.DiscordRestClient):
        def __init__(self) -> None:
            super().__init__("token")
            self.calls: list[dict[str, str] | None] = []

        def _request_json(self, path: str, params: dict[str, str] | None = None) -> object:
            assert path == "/channels/channel/messages"
            self.calls.append(params)
            if params and params.get("before") == "901":
                return [{"id": "900"}, {"id": "899"}]
            return [{"id": str(message_id)} for message_id in range(1000, 900, -1)]

    client = FakeClient()
    messages = client.messages_since("channel", "900")

    assert [message["id"] for message in messages] == [str(message_id) for message_id in range(899, 1001)]
    assert client.calls == [{"limit": "100"}, {"limit": "100", "before": "901"}]


@pytest.mark.unit
def test_message_fetch_stops_at_the_boundary_when_the_cursor_was_deleted() -> None:
    class FakeClient(discord_sync.DiscordRestClient):
        def __init__(self) -> None:
            super().__init__("token")
            self.calls: list[dict[str, str] | None] = []

        def _request_json(self, path: str, params: dict[str, str] | None = None) -> object:
            assert path == "/channels/channel/messages"
            self.calls.append(params)
            if params and params.get("before") == "901":
                return [{"id": str(message_id)} for message_id in range(899, 799, -1)]
            if params and params.get("before") == "800":
                raise AssertionError("a deleted cursor must not trigger a full-history scan")
            return [{"id": str(message_id)} for message_id in range(1000, 900, -1)]

    client = FakeClient()
    messages = client.messages_since("channel", "900")

    identifiers = [message["id"] for message in messages]
    assert len(identifiers) == 200
    assert identifiers[0] == "800"
    assert identifiers[-1] == "1000"
    assert "900" not in identifiers
    assert client.calls == [{"limit": "100"}, {"limit": "100", "before": "901"}]


@pytest.mark.unit
def test_sync_rebuilds_a_channel_for_new_messages_and_advances_cursor_afterward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = {
        "id": "10",
        "type": 0,
        "timestamp": "2026-10-21T01:02:03Z",
        "author": {"username": "Ada"},
        "content": "VM discussion",
    }
    stored = DiscordSourceMessage(
        channel_id="channel",
        message_id="10",
        channel_name="engineering",
        author_name="Ada",
        message_text="VM discussion",
        sent_at=datetime(2026, 10, 21, 1, 2, 3, tzinfo=UTC),
    )

    class FakeClient:
        def channel_name(self, channel_id: str) -> str:
            assert channel_id == "channel"
            return "engineering"

        def messages_since(self, channel_id: str, cursor: str | None) -> list[dict[str, object]]:
            assert (channel_id, cursor) == ("channel", "9")
            return [message]

    calls: list[object] = []
    monkeypatch.setattr(discord_sync, "initialize_database", lambda: calls.append("initialize"))
    monkeypatch.setattr(discord_sync, "discord_sync_cursor", lambda _: "9")
    monkeypatch.setattr(discord_sync, "upsert_discord_source_messages", lambda rows: len(rows))
    monkeypatch.setattr(discord_sync, "discord_reindex_required", lambda _: False)
    monkeypatch.setattr(discord_sync, "discord_source_messages", lambda _: [stored])
    monkeypatch.setattr(
        discord_sync,
        "ingest_units",
        lambda units, **kwargs: calls.append((list(units), kwargs)) or {"chunks": 1},
    )
    monkeypatch.setattr(
        discord_sync,
        "set_discord_sync_cursor",
        lambda *args: calls.append(("cursor", args)),
    )
    monkeypatch.setattr(discord_sync, "create_indexes", lambda: calls.append("indexes"))

    reports = discord_sync.sync_channels(FakeClient(), ["channel"])

    assert reports[0].reindexed is True
    assert reports[0].chunks == 1
    assert calls[-2] == ("cursor", ("channel", "engineering", "10"))
    assert calls[-1] == "indexes"
