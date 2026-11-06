"""Synchronize selected Discord channels directly into the retrieval corpus.

The connector stores normalized, redacted messages in PostgreSQL, not raw API
payloads. When a channel has new or edited messages, it safely rebuilds that
channel's chunks and then records the Discord cursor. It intentionally does
not write Discord exports to the repository or filesystem.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import DISCORD_API_BASE_URL, DISCORD_INGEST_BOT_TOKEN, DISCORD_INGEST_CHANNEL_IDS
from .ingestion.clean import RawUnit, discord_units_from_messages
from .ingestion.pipeline import ingest_units
from .storage.postgres import (
    DiscordSourceMessage,
    create_indexes,
    discord_reindex_required,
    discord_source_messages,
    discord_sync_cursor,
    initialize_database,
    set_discord_sync_cursor,
    upsert_discord_source_messages,
)


class DiscordApiError(RuntimeError):
    """A Discord API request could not be completed."""


class DiscordRestClient:
    """Small REST client for the endpoints needed by the scheduled sync."""

    def __init__(self, token: str, *, api_base_url: str = DISCORD_API_BASE_URL) -> None:
        self._token = token
        self._api_base_url = api_base_url.rstrip("/")

    def _request_json(self, path: str, params: dict[str, str] | None = None) -> Any:
        query = f"?{urlencode(params)}" if params else ""
        request = Request(
            f"{self._api_base_url}{path}{query}",
            headers={
                "Authorization": f"Bot {self._token}",
                "Accept": "application/json",
                "User-Agent": "benchmark-cli-tool discord-ingestion",
            },
        )
        for attempt in range(3):
            try:
                with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed Discord API URL
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                retry_after = 0.0
                try:
                    retry_after = float(json.loads(body).get("retry_after", 0))
                except (ValueError, json.JSONDecodeError):
                    pass
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise DiscordApiError(f"Discord API request failed with HTTP {exc.code}.") from exc
                time.sleep(max(retry_after, 2**attempt))
            except URLError as exc:
                if attempt == 2:
                    raise DiscordApiError("Discord API request could not be reached.") from exc
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def channel_name(self, channel_id: str) -> str:
        payload = self._request_json(f"/channels/{channel_id}")
        if not isinstance(payload, dict):
            raise DiscordApiError(f"Discord returned an invalid channel response for {channel_id}.")
        return str(payload.get("name") or channel_id)

    def messages_since(self, channel_id: str, cursor: str | None) -> list[dict[str, Any]]:
        """Fetch newest-first pages until a page reaches the saved cursor.

        The boundary page is deliberately included. It provides a small overlap
        that detects edits to recent messages. Stopping once a page passes the
        cursor also bounds the work when the cursor message was deleted.
        """
        before: str | None = None
        messages_by_id: dict[str, dict[str, Any]] = {}
        while True:
            params = {"limit": "100"}
            if before:
                params["before"] = before
            payload = self._request_json(f"/channels/{channel_id}/messages", params)
            if not isinstance(payload, list):
                raise DiscordApiError(f"Discord returned an invalid message response for {channel_id}.")
            page = [message for message in payload if isinstance(message, dict) and str(message.get("id", "")).isdigit()]
            for message in page:
                messages_by_id[str(message["id"])] = message
            if cursor and cursor.isdigit() and any(int(message["id"]) <= int(cursor) for message in page):
                break
            if len(payload) < 100 or not page:
                break
            before = str(page[-1]["id"])
        return [messages_by_id[message_id] for message_id in sorted(messages_by_id, key=int)]


def _source_filename(channel_id: str) -> str:
    return f"discord-channel-{channel_id}.json"


def _newest_message_id(messages: Sequence[dict[str, Any]], previous: str | None) -> str | None:
    identifiers = [previous] if previous and previous.isdigit() else []
    identifiers.extend(str(message["id"]) for message in messages if str(message.get("id", "")).isdigit())
    return str(max(map(int, identifiers))) if identifiers else None


def _has_new_message(messages: Sequence[dict[str, Any]], cursor: str | None) -> bool:
    if cursor is None:
        return bool(messages)
    if not cursor.isdigit():
        return True
    return any(str(message.get("id", "")).isdigit() and int(message["id"]) > int(cursor) for message in messages)


def _source_messages(units: Sequence[RawUnit], channel_id: str, channel_name: str) -> list[DiscordSourceMessage]:
    messages: list[DiscordSourceMessage] = []
    for unit in units:
        message_id = str(unit.extra.get("message_id", ""))
        if not message_id or unit.timestamp is None:
            continue
        edited = unit.extra.get("edited_timestamp")
        edited_at = datetime.fromisoformat(str(edited).replace("Z", "+00:00")) if edited else None
        messages.append(
            DiscordSourceMessage(
                channel_id=channel_id,
                message_id=message_id,
                channel_name=channel_name,
                author_name=unit.speaker or "unknown",
                message_text=unit.text,
                sent_at=unit.timestamp,
                edited_at=edited_at,
                reply_to_message_id=unit.extra.get("reply_to"),
            )
        )
    return messages


def _stored_units(messages: Sequence[DiscordSourceMessage]) -> list[RawUnit]:
    return [
        RawUnit(
            message.message_text,
            message.author_name,
            message.sent_at,
            ordinal,
            {
                "channel": message.channel_name,
                "section": message.channel_name,
                "message_id": message.message_id,
                "reply_to": message.reply_to_message_id,
                "edited_timestamp": message.edited_at.isoformat() if message.edited_at else None,
            },
        )
        for ordinal, message in enumerate(messages)
    ]


@dataclass(frozen=True, slots=True)
class ChannelSyncReport:
    channel_id: str
    channel_name: str
    fetched: int
    retained: int
    changed: int
    reindexed: bool
    chunks: int


def sync_channels(
    client: DiscordRestClient,
    channel_ids: Sequence[str],
    *,
    batch_size: int = 128,
) -> list[ChannelSyncReport]:
    """Synchronize every configured channel and return one operational report each."""
    initialize_database()
    reports: list[ChannelSyncReport] = []
    for channel_id in channel_ids:
        cursor = discord_sync_cursor(channel_id)
        channel_name = client.channel_name(channel_id)
        payload = client.messages_since(channel_id, cursor)
        units = discord_units_from_messages(payload, channel=channel_name)
        incoming = _source_messages(units, channel_id, channel_name)
        changed = upsert_discord_source_messages(incoming)
        reindex = discord_reindex_required(channel_id) or (
            bool(incoming) and _has_new_message(payload, cursor)
        )
        chunks = 0
        if reindex:
            report = ingest_units(
                _stored_units(discord_source_messages(channel_id)),
                source_filename=_source_filename(channel_id),
                source_type="discord",
                dry_run=False,
                replace_existing=True,
                batch_size=batch_size,
            )
            chunks = int(report["chunks"])
        # This follows successful message persistence and, where needed,
        # successful indexing. If embedding fails, the cursor remains behind
        # and a later run retries the same new range.
        set_discord_sync_cursor(channel_id, channel_name, _newest_message_id(payload, cursor))
        reports.append(
            ChannelSyncReport(channel_id, channel_name, len(payload), len(incoming), changed, reindex, chunks)
        )
    create_indexes()
    return reports


def _channel_ids(value: str | None) -> tuple[str, ...]:
    return tuple(channel_id.strip() for channel_id in (value or "").split(",") if channel_id.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", help="Comma-separated Discord channel IDs. Defaults to DISCORD_INGEST_CHANNEL_IDS.")
    parser.add_argument("--batch-size", type=int, default=128, help="Embedding inputs per request (1-128).")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 128:
        raise SystemExit("--batch-size must be between 1 and 128.")
    channel_ids = _channel_ids(args.channels) if args.channels is not None else DISCORD_INGEST_CHANNEL_IDS
    if not DISCORD_INGEST_BOT_TOKEN:
        raise SystemExit("DISCORD_INGEST_BOT_TOKEN is not configured.")
    if not channel_ids:
        raise SystemExit("Supply --channels or configure DISCORD_INGEST_CHANNEL_IDS.")
    reports = sync_channels(
        DiscordRestClient(DISCORD_INGEST_BOT_TOKEN), channel_ids, batch_size=args.batch_size
    )
    print(json.dumps([asdict(report) for report in reports], indent=2))


if __name__ == "__main__":
    main()
