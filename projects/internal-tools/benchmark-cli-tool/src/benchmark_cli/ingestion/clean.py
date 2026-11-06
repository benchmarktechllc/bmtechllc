"""Source parsers plus normalization and credential redaction for ingestion."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

_BOT_NAME = re.compile(r"(?:bot|webhook|github|ci|jenkins|dependabot)$", re.IGNORECASE)
_URL_ONLY = re.compile(r"^https?://\S+$", re.IGNORECASE)
_TIMESTAMP_GUTTER = re.compile(
    r"(?:^|\s)(?:\[?\d{1,2}:\d{2}:\d{2}(?:\.\d+)?\]?\s*(?:-->)?|\(\d{1,2}:\d{2}\))(?:\s|$)"
)
_FILLER = re.compile(r"^(?:um+|uh+|\[inaudible\]|\[crosstalk\]|\[silence\])(?:\s+|$)", re.IGNORECASE)
_MENTION = re.compile(r"<@!?(\d+)>")
_CHANNEL = re.compile(r"<#(\d+)>")
_EMOJI = re.compile(r"<a?:([\w-]+):\d+>")
_BLANKS = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")
_SPACE = re.compile(r"[ \t]+")
_PUNCTUATION_ONLY = re.compile(r"^[\W_]+$", re.UNICODE)
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\b(?:bearer\s+)[A-Za-z0-9._~+/=-]{12,}\b", re.IGNORECASE),
    re.compile(r"-----BEGIN(?: [A-Z]+)? PRIVATE KEY-----.*?-----END(?: [A-Z]+)? PRIVATE KEY-----", re.DOTALL),
    re.compile(r"\bpassword\s*=\s*[^\s;]+", re.IGNORECASE),
)
_SECTION_PATTERN = re.compile(r"^\s*(?:\d+[.)]\s*)?([^:]{2,100}):(?:\s|$)")
_NUMBERED_HEADING_PATTERN = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+(.{2,100})\s*$")
_DISCORD_CHANNEL_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+#(?P<channel>[\w-]+)\s*$")
_UTC_OFFSET = re.compile(r"\bUTC\s*(?P<sign>[+-])\s*(?P<hours>\d{1,2})(?::?(?P<minutes>\d{2}))?\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class RawUnit:
    """One atomic source unit that chunking preserves unless it is oversize."""

    text: str
    speaker: str | None
    timestamp: datetime | None
    ordinal: int
    extra: dict[str, Any]


def _section_blocks(text: str) -> list[tuple[str, int, int]]:
    """Return heading-delimited line ranges for a policy document."""
    starts: list[tuple[str, int]] = []
    lines = text.splitlines()
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        title: str | None = None
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
        elif match := _SECTION_PATTERN.match(stripped):
            title = match.group(1).strip()
        elif match := _NUMBERED_HEADING_PATTERN.match(stripped):
            title = match.group(1).strip()
        elif number == 1 and stripped:
            title = stripped.rstrip(":")
        if title:
            starts.append((title, number))
    return [
        (title, start, starts[index + 1][1] - 1 if index + 1 < len(starts) else len(lines))
        for index, (title, start) in enumerate(starts)
    ]


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    normalized = str(value).strip()
    normalized = re.sub(r"\s+UTC$", "+00:00", normalized, flags=re.IGNORECASE)
    try:
        return datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        for pattern in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(normalized, pattern)
            except ValueError:
                continue
        return None


def _discord_export_timezone(text: str) -> timezone | None:
    """Read a UTC offset declared in a text Discord export's preamble."""
    match = _UTC_OFFSET.search(text)
    if match is None:
        return None
    minutes = int(match.group("hours")) * 60 + int(match.group("minutes") or 0)
    if match.group("sign") == "-":
        minutes = -minutes
    return timezone(timedelta(minutes=minutes))


def _meeting_date(text: str) -> Date | None:
    """Extract a dated meeting header for text transcripts with clock-only turns."""
    for line in text.splitlines():
        match = re.match(r"\s*date\s*:\s*(?P<value>.+)", line, re.IGNORECASE)
        if match and (parsed := _parse_datetime(match.group("value"))) is not None:
            return parsed.date()
    return None


def _timestamp_for_turn(value: Any, meeting_date: Date | None) -> datetime | None:
    """Parse an absolute timestamp or attach a meeting date to a clock-only turn."""
    parsed = _parse_datetime(value)
    if parsed is not None:
        return parsed
    if meeting_date is None or not value:
        return None
    try:
        return datetime.combine(meeting_date, time.fromisoformat(str(value).strip()))
    except ValueError:
        return None


def normalize_text(text: str) -> str:
    """Normalize presentation noise and redact secrets before text reaches the API."""
    normalized = unicodedata.normalize("NFKC", text).replace("\ufeff", "").replace("\u200b", "")
    normalized = normalized.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"}))
    for pattern in _SECRET_PATTERNS:
        normalized = pattern.sub("[REDACTED]", normalized)

    in_fence = False
    lines: list[str] = []
    for line in normalized.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        line = line.rstrip()
        if not in_fence:
            line = _SPACE.sub(" ", line)
        if line.strip() and _PUNCTUATION_ONLY.match(line.strip()):
            continue
        lines.append(line)
    return _BLANKS.sub("\n\n", "\n".join(lines)).strip()


def _rewrite_discord(text: str, users: dict[str, str], channels: dict[str, str]) -> str:
    text = _MENTION.sub(lambda match: f"@{users.get(match.group(1), 'user')}", text)
    text = _CHANNEL.sub(lambda match: f"#{channels.get(match.group(1), 'channel')}", text)
    return _EMOJI.sub(lambda match: f":{match.group(1)}:", text)


def _discord_messages(payload: Any) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str]]:
    if isinstance(payload, list):
        return payload, {}, {}
    if not isinstance(payload, dict):
        return [], {}, {}
    messages = payload.get("messages", payload.get("Messages", []))
    users = payload.get("users", payload.get("Users", []))
    channels = payload.get("channels", payload.get("Channels", []))
    user_lookup = {
        str(item.get("id", item.get("Id"))): str(item.get("name", item.get("username", item.get("Name", "user"))))
        for item in users if isinstance(item, dict)
    }
    channel_lookup = {
        str(item.get("id", item.get("Id"))): str(item.get("name", item.get("Name", "channel")))
        for item in channels if isinstance(item, dict)
    }
    return messages if isinstance(messages, list) else [], user_lookup, channel_lookup


def discord_units_from_messages(
    messages: list[dict[str, Any]],
    *,
    channel: str,
    users: dict[str, str] | None = None,
    channels: dict[str, str] | None = None,
) -> list[RawUnit]:
    """Normalize Discord API/export messages into ingestion units.

    This is deliberately shared by file exports and the live connector so both
    paths redact secrets and apply the same bot/content filtering rules.
    """
    user_lookup = users or {}
    channel_lookup = channels or {}
    units: list[RawUnit] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        author = message.get("author", message.get("Author", {})) or {}
        if not isinstance(author, dict):
            continue
        author_name = str(
            author.get("global_name")
            or author.get("name")
            or author.get("username")
            or author.get("Name")
            or "unknown"
        )
        if author.get("isBot") or author.get("bot") or _BOT_NAME.search(author_name):
            continue
        kind = str(message.get("type", message.get("Type", "Default")))
        if kind.casefold() not in {"default", "reply", "0", "19"}:
            continue
        content = _rewrite_discord(
            str(message.get("content", message.get("Content", ""))), user_lookup, channel_lookup
        )
        content = normalize_text(content)
        if len(content) < 3 or _URL_ONLY.fullmatch(content):
            continue
        reference = message.get("message_reference", message.get("reference", message.get("replyTo")))
        reply_to: str | None
        if isinstance(reference, dict):
            value = reference.get("message_id", reference.get("messageId"))
            reply_to = str(value) if value is not None else None
        else:
            reply_to = str(reference) if reference is not None else None
        units.append(
            RawUnit(
                content,
                author_name,
                _parse_datetime(message.get("timestamp", message.get("Timestamp"))),
                len(units),
                {
                    "channel": str(message.get("channel", message.get("channelName", channel))),
                    "section": str(message.get("channel", message.get("channelName", channel))),
                    "message_id": str(message.get("id", message.get("Id", ""))),
                    "reply_to": reply_to,
                    "edited_timestamp": message.get("edited_timestamp", message.get("editedTimestamp")),
                },
            )
        )
    return units


def parse_discord(path: Path) -> Iterator[RawUnit]:
    """Parse DiscordChatExporter JSON or a lightweight timestamped TXT export."""
    if path.suffix.casefold() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        messages, users, channels = _discord_messages(payload)
        channel = str(payload.get("channel", payload.get("channelName", path.stem))) if isinstance(payload, dict) else path.stem
        yield from discord_units_from_messages(messages, channel=channel, users=users, channels=channels)
        return

    source_text = path.read_text(encoding="utf-8")
    export_timezone = _discord_export_timezone(source_text)
    channel = path.stem
    channel_boundary = False
    for ordinal, line in enumerate(source_text.splitlines()):
        if heading := _DISCORD_CHANNEL_HEADING.match(line):
            channel = heading.group("channel")
            channel_boundary = True
            continue
        # Discord's text export prefixes replies with tree glyphs and commonly
        # includes an author role/id between the timestamp and the colon.
        normalized_line = re.sub(r"^\s*(?:[└├│─]+\s*)+", "", line)
        match = re.match(
            r"(?:\[(?P<time>[^]]+)\]\s*)?"
            r"(?P<speaker>[^:\[][^:]{0,120}?)(?:\s*\[[^]]+\])?:\s*(?P<text>.*)",
            normalized_line,
        )
        if not match:
            continue
        speaker = match.group("speaker").strip()
        content = normalize_text(match.group("text"))
        if _BOT_NAME.search(speaker) or len(content) < 3 or _URL_ONLY.fullmatch(content):
            continue
        timestamp = _parse_datetime(match.group("time"))
        if timestamp is not None and timestamp.tzinfo is None and export_timezone is not None:
            timestamp = timestamp.replace(tzinfo=export_timezone)
        yield RawUnit(
            content,
            speaker,
            timestamp,
            ordinal,
            {"channel": channel, "section": channel, "boundary": channel_boundary},
        )
        channel_boundary = False


def parse_transcript(path: Path) -> Iterator[RawUnit]:
    """Parse JSON segments or Speaker: text transcript exports and merge speaker turns."""
    if path.suffix.casefold() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        segments = payload.get("segments", payload) if isinstance(payload, dict) else payload
        meeting = str(payload.get("meeting", path.stem)) if isinstance(payload, dict) else path.stem
        meeting_date = _parse_datetime(payload.get("date")).date() if isinstance(payload, dict) and _parse_datetime(payload.get("date")) else None
    else:
        source_text = path.read_text(encoding="utf-8")
        meeting_date = _meeting_date(source_text)
        segments = []
        pending_turn: dict[str, Any] | None = None
        for line in source_text.splitlines():
            match = re.match(
                r"(?:\[(?P<time>[^]]+)\]\s*)?"
                r"(?P<speaker>[^:\[]{1,80}):\s*(?P<text>.*)",
                line,
            )
            if match:
                if pending_turn is not None:
                    segments.append(pending_turn)
                    pending_turn = None
                turn = {
                    "speaker": match.group("speaker").strip(),
                    "timestamp": match.group("time"),
                    "text": match.group("text"),
                }
                if str(turn["text"]).strip():
                    segments.append(turn)
                elif turn["timestamp"] is not None:
                    # Some transcript exports put a diarized speaker heading on
                    # one line and the corresponding speech on the next.
                    pending_turn = turn
            elif pending_turn is not None and line.strip():
                pending_turn["text"] = line
                segments.append(pending_turn)
                pending_turn = None
            else:
                segments.append({"text": line})
        if pending_turn is not None:
            segments.append(pending_turn)
        meeting = path.stem
    if not isinstance(segments, list):
        return
    section: str | None = None
    pending_speaker: str | None = None
    pending_text: list[str] = []
    pending_time: datetime | None = None
    ordinal = 0

    def flush() -> RawUnit | None:
        nonlocal ordinal, pending_speaker, pending_text, pending_time
        if not pending_text:
            return None
        unit = RawUnit(" ".join(pending_text), pending_speaker, pending_time, ordinal, {"meeting": meeting, "section": section})
        ordinal += 1
        pending_speaker, pending_text, pending_time = None, [], None
        return unit

    for segment in segments:
        if not isinstance(segment, dict):
            continue
        raw = str(segment.get("text", segment.get("content", "")))
        heading = raw.strip().rstrip(":")
        if heading.startswith("#") or re.fullmatch(r"(?:agenda|topic|section)\s*[:\-].+", heading, re.IGNORECASE):
            flushed = flush()
            if flushed:
                yield flushed
            section = heading.lstrip("#").strip()
            continue
        raw = _TIMESTAMP_GUTTER.sub(" ", raw)
        text = normalize_text(raw)
        if not text or _FILLER.fullmatch(text):
            continue
        speaker = str(segment.get("speaker", segment.get("name", "Unknown"))).strip()
        timestamp = _timestamp_for_turn(
            segment.get("timestamp", segment.get("start")), meeting_date
        )
        if pending_speaker and speaker != pending_speaker:
            flushed = flush()
            if flushed:
                yield flushed
        if not pending_speaker:
            pending_speaker, pending_time = speaker, timestamp
        pending_text.append(text)
    flushed = flush()
    if flushed:
        yield flushed


def parse_policy_doc(path: Path) -> Iterator[RawUnit]:
    """Parse policy documents into paragraph units carrying byte-compatible section names."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    blocks = _section_blocks(text) or [(path.stem, 1, len(lines))]
    ordinal = 0
    for section, start, end in blocks:
        content = "\n".join(lines[start - 1 : end]).strip()
        for paragraph in re.split(r"\n\s*\n", content):
            normalized = normalize_text(paragraph)
            if not normalized:
                continue
            yield RawUnit(normalized, None, None, ordinal, {"section": section, "boundary": ordinal == 0 or paragraph == content})
            ordinal += 1


def parse_by_type(path: Path) -> Iterator[RawUnit]:
    """Choose a parser from filename/source-directory conventions."""
    lowered = path.name.casefold()
    if "discord" in lowered or path.parent.name.casefold() == "discord":
        yield from parse_discord(path)
    elif "transcript" in lowered or path.parent.name.casefold() in {"transcripts", "meetings"}:
        yield from parse_transcript(path)
    else:
        yield from parse_policy_doc(path)
