"""Post a formatted Fireflies meeting summary to a Discord channel.

The Ask bot polls Fireflies on a timer while ``MEETING_NOTES_AUTOPOST`` is on
(``1``, ``true`` or ``yes``). ``MEETING_NOTES_CHANNEL_ID`` names the channel.
``MEETING_NOTES_POLL_SECONDS`` sets the interval (default 600). The feature is
off by default and the bot still answers questions when it is off.

One list call runs per tick. Detail is fetched only for meetings recorded at or
after the stored watermark that are not yet posted and not live or silent. The
first enable sets the watermark to now, so nothing older is posted. The
watermark never moves past the earliest meeting that is still processing.
``meeting_notes_posts`` has the transcript id as its primary key, and a post is
recorded only after Discord accepted it.

``benchmark-meeting-notes`` prints a summary for one transcript, or for every
transcript since a date, and never posts or writes to the database.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import discord

from .discord_ask import NO_MENTIONS, AskConfig, split_discord_messages
from .fireflies_sync import (
    FirefliesApiError,
    FirefliesGraphQLClient,
    FirefliesMeeting,
    pending_reason,
)

logger = logging.getLogger(__name__)

NOTES_LIMIT = 1900
DISCORD_LIMIT = 2000
_TRANSCRIPT_CHARACTER_CAP = 400_000
_DISCUSSION = "Discussion"
_TODOS = "ToDos"
_NEW_YORK = ZoneInfo("America/New_York")
_BULLET = re.compile(r"^\s*(?:[-*•–·]|\d+[.)])\s*")
_HEADER = re.compile(r"^[\s#*_`>]*(discussion|todos?|to-dos?|to dos?)[\s:*_`]*$", re.IGNORECASE)
_URL_LINE = re.compile(r"^\s*<?https?://\S+>?\s*$")
_TYPOGRAPHIC_DASHES = str.maketrans({"\u2011": "-", "\u2013": "-", "\u2014": "-"})

SYSTEM_PROMPT = f"""You write a Discord meeting recap from a meeting transcript.
Output ONLY the Discord message text, with no preamble and no code fences, in exactly this shape:

**<Month D, YYYY>**

{_DISCUSSION}

• short topic line
• ...

{_TODOS}

• action item (Owner)
• ...

Rules:
- The first line is the date given to you, bold, exactly as provided.
- The section headers are exactly "{_DISCUSSION}" and "{_TODOS}" with no markdown.
- Every item starts with "• " (bullet and a space) and is one concise line.
- Do not invent facts. Do not list ticket or card dumps. Do not @ anyone.
- Use only plain ASCII hyphen-minus "-" in bullet text. Never use non-breaking hyphens, en-dashes
  or em-dashes.
- Stay under 1500 characters in total. If it is long, drop the least important Discussion items
  first and keep the ToDos.

Discussion rules:
- Short topic lines, not dense sentences. Each bullet is one concise phrase (roughly one clause),
  similar in length and style to a heading, not a paragraph.

ToDos rules:
- Prefer the concrete action items that were actually agreed in the meeting. Do not invent tasks.
- Put an owner in parentheses ONLY when someone in the spoken transcript explicitly took the item
  or was assigned it. If no owner was clearly assigned, omit the parentheses entirely.
- Use first names only, never full legal names or last names.
- One agreed item is one bullet. Never give the same item to several people as separate bullets and
  never split one item into a bullet per person. If a joint action clearly involves two named
  people, keep it as one bullet and name both in the sentence.
- A joint action involving two named people is one ToDo written as "X and Y to ..." using first
  names, never two bullets.
- Derive ToDos and owners only from the spoken transcript. Any Fireflies hint is not a source of
  action items or owners.
- Mark optional items "optional".
- Do not write a link. It is added afterwards.
- If there are no action items, write "• None" under {_TODOS}."""


class NotesStore(Protocol):
    def watermark(self) -> datetime | None: ...
    def set_watermark(self, value: datetime) -> None: ...
    def posted(self, transcript_id: str) -> bool: ...
    def record(
        self, transcript_id: str, channel_id: str, message_ids: Sequence[str], title: str
    ) -> bool: ...


class PostgresNotesStore:
    def watermark(self) -> datetime | None:
        from .storage.postgres import meeting_notes_watermark

        return meeting_notes_watermark()

    def set_watermark(self, value: datetime) -> None:
        from .storage.postgres import set_meeting_notes_watermark

        set_meeting_notes_watermark(value)

    def posted(self, transcript_id: str) -> bool:
        from .storage.postgres import meeting_notes_posted

        return meeting_notes_posted(transcript_id)

    def record(
        self, transcript_id: str, channel_id: str, message_ids: Sequence[str], title: str
    ) -> bool:
        from .storage.postgres import record_meeting_notes_post

        return record_meeting_notes_post(transcript_id, channel_id, message_ids, title)


@dataclass(frozen=True, slots=True)
class TickResult:
    posted: int = 0
    pending: int = 0
    skipped: int = 0
    initialized: bool = False
    watermark: datetime | None = None


Summarizer = Callable[[FirefliesMeeting], str]
Sender = Callable[[list[str]], Awaitable[list[str]]]


def should_start_meeting_notes(config: AskConfig, api_key: str | None) -> bool:
    """Whether the polling loop may start. Logs why it will not."""
    if not config.meeting_notes_enabled:
        return False
    if config.meeting_notes_channel_id is None:
        logger.error("MEETING_NOTES_AUTOPOST is on but MEETING_NOTES_CHANNEL_ID is missing or invalid.")
        return False
    if not (api_key or "").strip():
        logger.error("MEETING_NOTES_AUTOPOST is on but FIREFLIES_API_KEY is missing.")
        return False
    return True


def as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def date_title(recorded_at: datetime | None, *, now: datetime | None = None) -> str:
    moment = recorded_at if recorded_at is not None else (now or datetime.now(UTC))
    local = as_utc(moment).astimezone(_NEW_YORK)
    return f"{local:%B} {local.day}, {local.year}"


def transcript_text(meeting: FirefliesMeeting) -> str:
    lines = [f"{turn.speaker}: {turn.text}" for turn in meeting.transcript]
    return "\n".join(lines)[:_TRANSCRIPT_CHARACTER_CAP]


def build_messages(meeting: FirefliesMeeting) -> list[dict[str, Any]]:
    parts = [
        f"Meeting title: {meeting.title}",
        f"Date line to use: **{date_title(meeting.recorded_at)}**",
    ]
    if meeting.overview:
        parts.append(
            "Optional high-level hint from Fireflies (may be wrong about details). The spoken "
            "transcript below is the source of truth for ToDos and owners:\n"
            f"{meeting.overview}"
        )
    parts.append(f"Transcript:\n{transcript_text(meeting)}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def generate_summary(meeting: FirefliesMeeting) -> str:
    """One meeting-notes-tier call, then deterministic formatting. Returns the full message."""
    from .config import MEETING_NOTES_REASONING_EFFORT
    from .llm import call_llm

    completion = call_llm(
        tier="meeting_notes",
        messages=build_messages(meeting),
        component="meeting_notes",
        reasoning_effort=MEETING_NOTES_REASONING_EFFORT,
    )
    raw = completion.choices[0].message.content or ""
    return normalize_summary(raw, meeting)


def _bullet_text(line: str) -> str:
    return _BULLET.sub("", line).strip()


def normalize_summary(
    raw: str, meeting: FirefliesMeeting, *, max_chars: int = NOTES_LIMIT
) -> str:
    """Rebuild the model output in the fixed layout and append the recording link."""
    section = ""
    discussion: list[str] = []
    todos: list[str] = []
    for line in raw.replace("```", "").splitlines():
        stripped = line.strip()
        if not stripped or _URL_LINE.match(stripped):
            continue
        header = _HEADER.match(stripped)
        if header:
            section = _DISCUSSION if header.group(1).casefold() == "discussion" else _TODOS
            continue
        if not section:
            continue
        text = _bullet_text(stripped).translate(_TYPOGRAPHIC_DASHES)
        if text:
            (discussion if section == _DISCUSSION else todos).append(f"• {text}")
    title = f"**{date_title(meeting.recorded_at)}**"
    link = f"<{meeting.url}>"

    def render(items: list[str]) -> str:
        todo_lines = todos or ["• None"]
        return "\n".join(
            [title, "", _DISCUSSION, "", *items, "", _TODOS, "", *todo_lines, "", link]
        )

    items = list(discussion)
    text = render(items)
    while len(text) > max_chars and items:
        items.pop()
        text = render(items)
    return text


def split_notes(text: str, *, limit: int = DISCORD_LIMIT) -> list[str]:
    return split_discord_messages(text, limit=limit)


def _is_silent(meeting: FirefliesMeeting) -> bool:
    return pending_reason(meeting) == "silent_meeting"


def _fetch_postable(
    client: FirefliesGraphQLClient, meeting: FirefliesMeeting
) -> tuple[FirefliesMeeting | None, bool]:
    """Return (loaded meeting, silent). A None meeting that is not silent is still pending."""
    detail = client.transcript(meeting.recording_id)
    if detail is None:
        return None, False
    if detail.silent:
        return None, True
    loaded = replace(
        meeting,
        transcript=detail.transcript,
        summary=detail.summary,
        overview=detail.overview,
    )
    if detail.live or pending_reason(loaded, sentences_known=True) is not None:
        return None, False
    return loaded, False


async def run_tick(
    client: FirefliesGraphQLClient,
    store: NotesStore,
    *,
    channel_id: str,
    summarize: Summarizer,
    send: Sender,
    now: datetime | None = None,
) -> TickResult:
    """One poll: post ready meetings at or after the watermark; hold it at the oldest pending one."""
    moment = as_utc(now or datetime.now(UTC))
    watermark = await asyncio.to_thread(store.watermark)
    if watermark is None:
        await asyncio.to_thread(store.set_watermark, moment)
        return TickResult(initialized=True, watermark=moment)
    watermark = as_utc(watermark)
    meetings = await asyncio.to_thread(client.list_meetings)
    candidates = sorted(
        (
            meeting
            for meeting in meetings
            if meeting.recorded_at is not None and as_utc(meeting.recorded_at) >= watermark
        ),
        key=lambda meeting: as_utc(meeting.recorded_at or moment),
    )
    held: datetime | None = None
    posted = pending = skipped = 0
    for meeting in candidates:
        recorded = as_utc(meeting.recorded_at or moment)
        if _is_silent(meeting):
            skipped += 1
            continue
        if await asyncio.to_thread(store.posted, meeting.recording_id):
            continue
        if pending_reason(meeting) is not None:
            held = held or recorded
            pending += 1
            continue
        try:
            loaded, silent = await asyncio.to_thread(_fetch_postable, client, meeting)
        except FirefliesApiError:
            logger.error("Fireflies detail request failed for %s.", meeting.recording_id)
            held = held or recorded
            pending += 1
            continue
        if silent:
            skipped += 1
            continue
        if loaded is None:
            held = held or recorded
            pending += 1
            continue
        try:
            message = await asyncio.to_thread(summarize, loaded)
            parts = split_notes(message)
        except Exception:
            logger.exception("Meeting summary failed for %s.", meeting.recording_id)
            held = held or recorded
            pending += 1
            continue
        message_ids: list[str] = []
        try:
            message_ids = await send(parts)
        except Exception:
            logger.exception("Posting meeting notes failed for %s.", meeting.recording_id)
        if not message_ids:
            held = held or recorded
            pending += 1
            continue
        recorded_ok = await asyncio.to_thread(
            store.record, meeting.recording_id, channel_id, message_ids, meeting.title
        )
        if not recorded_ok:
            logger.warning("Meeting %s was already recorded as posted.", meeting.recording_id)
        posted += 1
        logger.info("Posted meeting notes for %s in %s message(s).", meeting.recording_id, len(parts))
    new_mark = watermark if held is None else max(held, watermark)
    if new_mark > watermark:
        await asyncio.to_thread(store.set_watermark, new_mark)
    return TickResult(posted, pending, skipped, False, new_mark)


class MeetingNotesPoller:
    """Runs ticks on an interval. The lock keeps ticks from overlapping."""

    def __init__(
        self,
        client: discord.Client,
        config: AskConfig,
        fireflies: FirefliesGraphQLClient,
        *,
        store: NotesStore | None = None,
        summarize: Summarizer = generate_summary,
    ) -> None:
        self._client = client
        self._config = config
        self._fireflies = fireflies
        self._store = store or PostgresNotesStore()
        self._summarize = summarize
        self._lock = asyncio.Lock()

    async def tick(self) -> TickResult | None:
        channel_id = self._config.meeting_notes_channel_id
        if not self._config.meeting_notes_enabled or channel_id is None:
            return None
        if self._lock.locked():
            return None
        async with self._lock:
            return await run_tick(
                self._fireflies,
                self._store,
                channel_id=str(channel_id),
                summarize=self._summarize,
                send=self._send,
            )

    async def _send(self, parts: list[str]) -> list[str]:
        channel_id = self._config.meeting_notes_channel_id
        assert channel_id is not None
        channel: Any = self._client.get_channel(channel_id)
        if channel is None:
            channel = await self._client.fetch_channel(channel_id)
        ids: list[str] = []
        try:
            for part in parts:
                message = await channel.send(part, allowed_mentions=NO_MENTIONS)
                ids.append(str(message.id))
        except Exception:
            if not ids:
                raise
            logger.exception("Meeting notes were only partly sent (%s of %s).", len(ids), len(parts))
        return ids

    async def run(self) -> None:
        await self._client.wait_until_ready()
        interval = max(self._config.meeting_notes_poll_seconds, 1)
        logger.info("Meeting notes autopost started, polling every %s seconds.", interval)
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Meeting notes tick failed.")
            await asyncio.sleep(interval)


def start_meeting_notes(
    client: discord.Client, api_key: str | None
) -> asyncio.Task[None] | None:
    config: AskConfig = client.config  # type: ignore[attr-defined]
    if not should_start_meeting_notes(config, api_key):
        return None
    from .config import FIREFLIES_API_BASE_URL

    fireflies = FirefliesGraphQLClient(str(api_key), api_base_url=FIREFLIES_API_BASE_URL)
    return asyncio.create_task(MeetingNotesPoller(client, config, fireflies).run())


def _parse_since(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    return as_utc(parsed)


def dry_run_summaries(
    client: FirefliesGraphQLClient,
    *,
    transcript_id: str | None,
    since: datetime | None,
    summarize: Summarizer = generate_summary,
) -> list[str]:
    """Formatted summaries only. Nothing is posted and the database is not touched."""
    if transcript_id is not None:
        found = client.transcript(transcript_id)
        if found is None:
            raise SystemExit("Transcript not found.")
        if pending_reason(found, sentences_known=True) is not None:
            raise SystemExit("Transcript is not ready (live, silent, or no sentences).")
        return [summarize(found)]
    results: list[str] = []
    for meeting in client.list_meetings():
        if since is not None and (
            meeting.recorded_at is None or as_utc(meeting.recorded_at) < since
        ):
            continue
        if pending_reason(meeting) is not None:
            continue
        loaded, _silent = _fetch_postable(client, meeting)
        if loaded is not None:
            results.append(summarize(loaded))
    return results


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript-id", help="Fireflies transcript id to summarize.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print the summary only. This is always on; the CLI never posts.",
    )
    parser.add_argument("--out", type=Path, help="Also write the summary to this file.")
    parser.add_argument("--backfill", action="store_true", help="Summarize every transcript.")
    parser.add_argument("--since", help="With --backfill, only meetings at or after this ISO 8601 time.")
    args = parser.parse_args(argv)
    if bool(args.transcript_id) == bool(args.backfill):
        raise SystemExit("Pass exactly one of --transcript-id or --backfill.")
    if args.since and not args.backfill:
        raise SystemExit("--since needs --backfill.")
    since: datetime | None = None
    if args.since:
        try:
            since = _parse_since(args.since)
        except ValueError:
            raise SystemExit("--since must be an ISO 8601 date or time.") from None
    api_key = os.getenv("FIREFLIES_API_KEY", "").strip()
    if not api_key or not os.getenv("OPENAI_API_KEY", "").strip():
        raise SystemExit("FIREFLIES_API_KEY and OPENAI_API_KEY are required.")
    from .config import FIREFLIES_API_BASE_URL

    try:
        summaries = dry_run_summaries(
            FirefliesGraphQLClient(api_key, api_base_url=FIREFLIES_API_BASE_URL),
            transcript_id=args.transcript_id,
            since=since,
        )
    except FirefliesApiError as exc:
        raise SystemExit(str(exc)) from None
    output = "\n\n----\n\n".join(summaries)
    sys.stdout.write(output + "\n")
    if args.out:
        args.out.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
