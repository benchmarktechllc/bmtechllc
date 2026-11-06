"""Synchronize Fireflies meeting transcripts into the retrieval corpus.

Lists transcripts visible to ``FIREFLIES_API_KEY``. A meeting is indexed once
Fireflies has a finalized transcript. Each meeting is one source,
``fireflies/<transcript_id>.md``, with ``source_type`` ``transcript`` and
``kind`` ``meeting_transcript``. The Fireflies transcript id is stored in the
``recording_id`` metadata field. The summary is stored when Fireflies returns
an overview, short summary, or action items. The source is replaced as a unit,
the same way a Fathom meeting is replaced. Fathom sync is left in place.

The list call does not request sentences or summary text. Free-plan keys allow
about 30 requests a minute and 500 a day, so detail is fetched only for
meetings whose listing changed. The list payload has no separate ``updated_at``.
A meeting is skipped when its title, transcript URL, conferencing link, host,
attendees, date, duration, and summary status still match the stored row. When
those change, a content hash skips the embedding call if the text is unchanged.

Live meetings (``is_live``), silent meetings (``silent_meeting``), and detail
payloads with no sentences are left untouched and retried next run. Meetings
that disappear from the accessible set are deleted from ``document_chunks``.
``--dry-run`` still calls Fireflies and counts chunks. It does not embed, write,
or delete.

HTTP 429 and GraphQL ``too_many_requests`` errors honor ``Retry-After``. A wait
longer than the per-minute window is surfaced instead of retried, so a spent
daily quota is not extended by more requests. A failed run stops before orphan
deletion.

``benchmark-sync-fireflies`` needs ``FIREFLIES_API_KEY``, ``DATABASE_URL``, and
``OPENAI_API_KEY``. The ``fireflies-ingestion`` workflow runs on the
``discord-ingestion`` GitHub environment. Never commit secrets.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import FIREFLIES_API_BASE_URL, FIREFLIES_API_KEY
from .ingestion.clean import RawUnit, normalize_text
from .ingestion.pipeline import content_hash, ingest_units
from .storage.postgres import (
    FirefliesSyncState,
    create_indexes,
    delete_fireflies_sync_state,
    delete_source,
    fireflies_sync_states,
    initialize_database,
    upsert_fireflies_sync_state,
)

FIREFLIES_SYNC_VERSION = "1"
_PAGE_SIZE = 50
_MAX_PAGES = 1000
_MAX_RATE_LIMIT_WAIT = 60.0
_EPOCH_MS = 10_000_000_000
_TRANSCRIPT_ID = re.compile(r"[A-Za-z0-9_-]{1,200}")
_CLOCK = re.compile(
    r"^(?:(?P<hours>\d+):)?(?P<minutes>\d{1,2}):(?P<seconds>\d{2})(?:\.\d+)?$"
)
_RETRYABLE_HTTP = {500, 502, 503, 504}
_RATE_LIMIT_CODES = {"too_many_requests", "rate_limited", "rate_limit_exceeded"}

_MEETING_FIELDS = """
    id
    title
    date
    dateString
    duration
    transcript_url
    meeting_link
    host_email
    organizer_email
    participants
    meeting_attendees { displayName email name }
    user { name email }
    meeting_info { silent_meeting summary_status }
    is_live
""".strip("\n")

_LIST_QUERY = f"""
query Transcripts($limit: Int!, $skip: Int!) {{
  transcripts(limit: $limit, skip: $skip) {{
{_MEETING_FIELDS}
  }}
}}
""".strip()

_DETAIL_QUERY = f"""
query Transcript($id: String!) {{
  transcript(id: $id) {{
{_MEETING_FIELDS}
    sentences {{ speaker_name text start_time end_time }}
    summary {{ overview action_items short_summary }}
  }}
}}
""".strip()


class FirefliesApiError(RuntimeError):
    """A Fireflies API request could not be completed."""


@dataclass(frozen=True, slots=True)
class FirefliesTurn:
    speaker: str
    text: str
    timestamp: str


@dataclass(frozen=True, slots=True)
class FirefliesMeeting:
    """One transcript from the list, plus sentences and summary when loaded."""

    recording_id: str
    title: str
    url: str
    meeting_url: str
    host: str
    attendees: tuple[str, ...]
    recorded_at: datetime | None
    duration_minutes: float | None
    summary_status: str
    silent: bool
    live: bool
    transcript: tuple[FirefliesTurn, ...] = ()
    summary: str = ""


@dataclass(frozen=True, slots=True)
class MeetingSyncReport:
    recording_id: str
    title: str
    source_file: str
    action: str
    chunks: int


class FirefliesGraphQLClient:
    """GraphQL client for the transcript list and one-transcript detail."""

    def __init__(self, api_key: str, *, api_base_url: str = FIREFLIES_API_BASE_URL) -> None:
        self._api_key = api_key
        self._api_base_url = api_base_url

    def list_meetings(self) -> list[FirefliesMeeting]:
        """Return every transcript the key can see. Sentences are not requested."""
        skip = 0
        by_id: dict[str, FirefliesMeeting] = {}
        order: list[str] = []
        for _page in range(_MAX_PAGES):
            data = self._graphql(_LIST_QUERY, {"limit": _PAGE_SIZE, "skip": skip})
            items = data.get("transcripts") if isinstance(data, dict) else None
            if not isinstance(items, list):
                raise FirefliesApiError("Fireflies returned an invalid transcripts response.")
            if not items:
                return [by_id[recording_id] for recording_id in order]
            added = 0
            for item in items:
                if not isinstance(item, dict):
                    continue
                meeting = parse_meeting(item)
                if meeting is None:
                    continue
                if meeting.recording_id not in by_id:
                    order.append(meeting.recording_id)
                    added += 1
                by_id[meeting.recording_id] = meeting
            if len(items) < _PAGE_SIZE:
                return [by_id[recording_id] for recording_id in order]
            if added == 0:
                raise FirefliesApiError("Fireflies transcript pagination repeated a page.")
            skip += len(items)
        raise FirefliesApiError("Fireflies transcript pagination did not finish.")

    def transcript(self, recording_id: str) -> FirefliesMeeting | None:
        """Return one transcript. A missing transcript is None, not an error."""
        if _transcript_id(recording_id) != recording_id:
            raise FirefliesApiError("Fireflies transcript id is invalid.")
        data = self._graphql(_DETAIL_QUERY, {"id": recording_id})
        item = data.get("transcript") if isinstance(data, dict) else None
        if item is None:
            return None
        if not isinstance(item, dict):
            raise FirefliesApiError("Fireflies returned an invalid transcript response.")
        meeting = parse_meeting(item)
        if meeting is None:
            raise FirefliesApiError("Fireflies returned an invalid transcript response.")
        return meeting

    def _graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
        request = Request(
            self._api_base_url,
            data=payload,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "benchmark-cli-tool fireflies-ingestion",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        for attempt in range(6):
            try:
                with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed Fireflies URL
                    raw = response.read()
                    headers = response.headers
            except HTTPError as exc:
                code = exc.code
                headers = exc.headers
                raw = _read_error_body(exc)
                exc.close()
                try:
                    _handle_http_failure(code, headers, raw, attempt)
                except FirefliesApiError as err:
                    raise FirefliesApiError(str(err)) from None
                continue
            except URLError:
                if attempt >= 2:
                    raise FirefliesApiError(
                        "Fireflies API request could not be reached."
                    ) from None
                time.sleep(min(2**attempt, _MAX_RATE_LIMIT_WAIT))
                continue
            body = _decode_json(raw)
            errors = body.get("errors")
            wait = _rate_limit_wait(errors, headers)
            if wait is not None:
                try:
                    time.sleep(_rate_limit_delay(wait, attempt))
                except FirefliesApiError as err:
                    raise FirefliesApiError(str(err)) from None
                continue
            if errors:
                raise FirefliesApiError("Fireflies API request failed.")
            data = body.get("data")
            if not isinstance(data, dict):
                raise FirefliesApiError("Fireflies returned an invalid GraphQL response.")
            return data
        raise FirefliesApiError("Fireflies API request failed.")


def parse_meeting(item: dict[str, Any]) -> FirefliesMeeting | None:
    """Normalize one transcript object. Items without a safe id are skipped."""
    recording_id = _transcript_id(item.get("id"))
    if recording_id is None:
        return None
    title = normalize_text(str(item.get("title") or "")) or "Untitled meeting"
    url = str(item.get("transcript_url") or "").strip()
    if not url:
        url = f"https://app.fireflies.ai/view/{recording_id}"
    info = item.get("meeting_info") if isinstance(item.get("meeting_info"), dict) else {}
    host = _host(item)
    return FirefliesMeeting(
        recording_id=recording_id,
        title=title,
        url=url,
        meeting_url=str(item.get("meeting_link") or "").strip(),
        host=host,
        attendees=_attendees(item, host),
        recorded_at=_recorded_at(item),
        duration_minutes=_duration_minutes(item.get("duration")),
        summary_status=_summary_status(info),
        silent=info.get("silent_meeting") is True,
        live=item.get("is_live") is True,
        transcript=_turns(item),
        summary=_summary(item),
    )


def pending_reason(meeting: FirefliesMeeting, *, sentences_known: bool = False) -> str | None:
    """Why a meeting should not be indexed, or None when detail can be used.

    The list payload does not include sentences. An empty transcript on that
    object is not ``no_sentences`` until ``sentences_known`` is set after detail.
    """
    if meeting.silent:
        return "silent_meeting"
    if meeting.live:
        return "not_ready"
    if sentences_known and not meeting.transcript:
        return "no_sentences"
    return None


def fireflies_source_file(recording_id: str) -> str:
    if _transcript_id(recording_id) != recording_id:
        raise FirefliesApiError("Fireflies transcript id is invalid.")
    return f"fireflies/{recording_id}.md"


def units_for_meeting(meeting: FirefliesMeeting) -> list[RawUnit]:
    """Build retrieval units for one meeting. Every unit carries the same provenance."""
    anchor = meeting.recorded_at or datetime(1970, 1, 1, tzinfo=UTC)
    provenance = {
        "meeting": meeting.title,
        "title": meeting.title,
        "url": meeting.url,
        "attendees": list(meeting.attendees),
        "recording_id": meeting.recording_id,
        "kind": "meeting_transcript",
    }
    units: list[RawUnit] = []
    ordinal = 0
    lines = [f"Meeting: {meeting.title}"]
    if meeting.recorded_at is not None:
        lines.append(f"Date: {meeting.recorded_at.date().isoformat()}")
    duration = _duration_text(meeting.duration_minutes)
    if duration:
        lines.append(f"Duration: {duration}")
    if meeting.url:
        lines.append(f"URL: {meeting.url}")
    if meeting.host:
        lines.append(f"Host: {meeting.host}")
    if meeting.attendees:
        lines.append("Attendees: " + ", ".join(meeting.attendees))
    if meeting.meeting_url:
        lines.append(f"Meeting link: {meeting.meeting_url}")
    units.append(
        RawUnit(
            "\n".join(lines),
            None,
            anchor,
            ordinal,
            {**provenance, "section": "Overview", "boundary": True},
        )
    )
    ordinal += 1
    if meeting.summary:
        units.append(
            RawUnit(
                f"Summary:\n{meeting.summary}",
                None,
                anchor,
                ordinal,
                {**provenance, "section": "Summary", "boundary": True},
            )
        )
        ordinal += 1
    for index, turn in enumerate(meeting.transcript):
        clock = turn.timestamp or "unknown time"
        units.append(
            RawUnit(
                f"{turn.speaker} ({clock}):\n{turn.text}",
                turn.speaker,
                _offset_timestamp(anchor, turn.timestamp),
                ordinal,
                {**provenance, "section": "Transcript", "boundary": index == 0},
            )
        )
        ordinal += 1
    return units


def meeting_digest(meeting: FirefliesMeeting, units: Sequence[RawUnit]) -> str:
    recorded = meeting.recorded_at
    lines = [
        FIREFLIES_SYNC_VERSION,
        meeting.recording_id,
        meeting.title,
        meeting.url,
        meeting.meeting_url,
        recorded.isoformat() if recorded else "",
        ",".join(meeting.attendees),
        *(unit.text for unit in units),
    ]
    return content_hash("\n".join(lines))


def sync_meetings(
    client: FirefliesGraphQLClient,
    *,
    dry_run: bool = False,
    batch_size: int = 128,
) -> list[MeetingSyncReport]:
    """Index finalized meetings and delete ones that left the accessible set."""
    meetings = client.list_meetings()
    if dry_run:
        return [_dry_run_report(client, meeting, batch_size) for meeting in meetings]

    initialize_database()
    stored = fireflies_sync_states()
    reports: list[MeetingSyncReport] = []
    seen: set[str] = set()
    for meeting in meetings:
        seen.add(meeting.recording_id)
        source = fireflies_source_file(meeting.recording_id)
        previous = stored.get(meeting.recording_id)
        if previous is not None and _listing_matches(previous, meeting):
            reports.append(
                MeetingSyncReport(meeting.recording_id, meeting.title, source, "unchanged", 0)
            )
            continue
        loaded = _load_content(client, meeting)
        if loaded is None:
            reports.append(
                MeetingSyncReport(meeting.recording_id, meeting.title, source, "pending", 0)
            )
            continue
        units = units_for_meeting(loaded)
        if not units:
            reports.append(
                MeetingSyncReport(meeting.recording_id, meeting.title, source, "pending", 0)
            )
            continue
        digest = meeting_digest(loaded, units)
        if (
            previous is not None
            and previous.content_sha256 == digest
            and previous.source_file == source
        ):
            _remember(loaded, source, digest)
            reports.append(
                MeetingSyncReport(meeting.recording_id, meeting.title, source, "unchanged", 0)
            )
            continue
        if previous is not None and previous.source_file != source:
            delete_source(previous.source_file)
        report = ingest_units(
            units,
            source_filename=source,
            source_type="transcript",
            dry_run=False,
            replace_existing=True,
            batch_size=batch_size,
        )
        _remember(loaded, source, digest)
        reports.append(
            MeetingSyncReport(
                meeting.recording_id, meeting.title, source, "indexed", int(report["chunks"])
            )
        )
    for recording_id, previous in stored.items():
        if recording_id in seen:
            continue
        delete_source(previous.source_file)
        delete_fireflies_sync_state(recording_id)
        reports.append(
            MeetingSyncReport(recording_id, previous.title, previous.source_file, "removed", 0)
        )
    create_indexes()
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Embedding inputs per request (1-128).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch meetings and count chunks without embedding or database writes.",
    )
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 128:
        raise SystemExit("--batch-size must be between 1 and 128.")
    if not FIREFLIES_API_KEY:
        raise SystemExit("FIREFLIES_API_KEY is required.")
    reports = sync_meetings(
        FirefliesGraphQLClient(FIREFLIES_API_KEY),
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    )
    print(json.dumps([asdict(report) for report in reports], indent=2))


def _dry_run_report(
    client: FirefliesGraphQLClient, meeting: FirefliesMeeting, batch_size: int
) -> MeetingSyncReport:
    source = fireflies_source_file(meeting.recording_id)
    loaded = _load_content(client, meeting)
    if loaded is None:
        return MeetingSyncReport(meeting.recording_id, meeting.title, source, "pending", 0)
    report = ingest_units(
        units_for_meeting(loaded),
        source_filename=source,
        source_type="transcript",
        dry_run=True,
        replace_existing=True,
        batch_size=batch_size,
    )
    return MeetingSyncReport(
        meeting.recording_id, meeting.title, source, "dry_run", int(report["chunks"])
    )


def _load_content(
    client: FirefliesGraphQLClient, meeting: FirefliesMeeting
) -> FirefliesMeeting | None:
    """Fetch sentences for a changed meeting. Listing fields stay the skip key."""
    if pending_reason(meeting) is not None:
        return None
    detail = client.transcript(meeting.recording_id)
    if detail is None or detail.silent or detail.live:
        return None
    loaded = replace(meeting, transcript=detail.transcript, summary=detail.summary)
    if pending_reason(loaded, sentences_known=True) is not None:
        return None
    return loaded


def _remember(meeting: FirefliesMeeting, source: str, digest: str) -> None:
    upsert_fireflies_sync_state(
        FirefliesSyncState(
            recording_id=meeting.recording_id,
            source_file=source,
            content_sha256=digest,
            title=meeting.title,
            fireflies_url=meeting.url,
            join_url=meeting.meeting_url,
            attendees=meeting.attendees,
            recorded_at=meeting.recorded_at,
            duration_minutes=meeting.duration_minutes,
            host=meeting.host,
            summary_status=meeting.summary_status,
        )
    )


def _listing_matches(previous: FirefliesSyncState, meeting: FirefliesMeeting) -> bool:
    return (
        previous.source_file == fireflies_source_file(meeting.recording_id)
        and previous.title == meeting.title
        and previous.fireflies_url == meeting.url
        and previous.join_url == meeting.meeting_url
        and previous.attendees == meeting.attendees
        and previous.host == meeting.host
        and previous.summary_status == meeting.summary_status
        and _same_optional_moment(previous.recorded_at, meeting.recorded_at)
        and _same_duration(previous.duration_minutes, meeting.duration_minutes)
    )


def _turns(item: dict[str, Any]) -> tuple[FirefliesTurn, ...]:
    if "sentences" not in item or item.get("sentences") is None:
        return ()
    sentences = item.get("sentences")
    if not isinstance(sentences, list):
        raise FirefliesApiError("Fireflies returned an invalid transcript response.")
    turns: list[FirefliesTurn] = []
    for sentence in sentences:
        if not isinstance(sentence, dict):
            continue
        speaker = normalize_text(str(sentence.get("speaker_name") or "Unknown")) or "Unknown"
        text = normalize_text(str(sentence.get("text") or ""))
        if not text:
            continue
        turns.append(FirefliesTurn(speaker, text, _format_clock(sentence.get("start_time"))))
    return tuple(turns)


def _summary(item: dict[str, Any]) -> str:
    if "summary" not in item or item.get("summary") is None:
        return ""
    summary = item.get("summary")
    if not isinstance(summary, dict):
        raise FirefliesApiError("Fireflies returned an invalid summary response.")
    overview = _summary_text(summary.get("overview"))
    short = _summary_text(summary.get("short_summary"))
    actions = _summary_text(summary.get("action_items"))
    parts: list[str] = []
    if overview:
        parts.append(overview)
    if short and short != overview:
        parts.append(short)
    if actions:
        if not actions.casefold().startswith("action items"):
            actions = f"Action items:\n{actions}"
        parts.append(actions)
    return "\n\n".join(parts)


def _summary_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        lines = [_summary_text(item) for item in value]
        return "\n".join(line for line in lines if line)
    if isinstance(value, dict):
        return ""
    return normalize_text(str(value))


def _host(item: dict[str, Any]) -> str:
    user = item.get("user") if isinstance(item.get("user"), dict) else {}
    for value in (
        user.get("name"),
        item.get("organizer_email"),
        item.get("host_email"),
        user.get("email"),
    ):
        label = normalize_text(str(value or ""))
        if label:
            return label
    return ""


def _attendees(item: dict[str, Any], host: str) -> tuple[str, ...]:
    names: list[str] = []
    seen_emails: set[str] = set()
    people = item.get("meeting_attendees")
    if isinstance(people, list):
        for person in people:
            if not isinstance(person, dict):
                continue
            email = normalize_text(str(person.get("email") or ""))
            label = normalize_text(
                str(person.get("displayName") or person.get("name") or email or "")
            )
            if label and label not in names:
                names.append(label)
            if email:
                seen_emails.add(email.casefold())
    participants = item.get("participants")
    if isinstance(participants, list):
        for person in participants:
            if not isinstance(person, str):
                continue
            label = normalize_text(person)
            if not label or label.casefold() in seen_emails or label in names:
                continue
            names.append(label)
            seen_emails.add(label.casefold())
    if host and host not in names:
        names.append(host)
    return tuple(dict.fromkeys(names))


def _summary_status(info: dict[str, Any]) -> str:
    return normalize_text(str(info.get("summary_status") or "")).casefold()


def _recorded_at(item: dict[str, Any]) -> datetime | None:
    parsed = _parse_time(item.get("dateString"))
    if parsed is not None:
        return parsed
    return _parse_epoch(item.get("date"))


def _duration_minutes(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")} or number < 0:
        return None
    return number


def _duration_text(minutes: float | None) -> str:
    if minutes is None:
        return ""
    rounded = round(float(minutes), 2)
    if abs(rounded - round(rounded)) < 1e-9:
        rendered = str(int(round(rounded)))
    else:
        rendered = f"{rounded:.2f}".rstrip("0").rstrip(".")
    return f"{rendered} min"


def _transcript_id(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    text = str(value).strip()
    if _TRANSCRIPT_ID.fullmatch(text) is None:
        return None
    return text


def _format_clock(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, (int, float)):
        return _clock_from_seconds(float(value))
    text = str(value).strip()
    if not text:
        return ""
    match = _CLOCK.fullmatch(text)
    if match:
        hours = int(match.group("hours") or 0)
        minutes = int(match.group("minutes"))
        seconds = int(match.group("seconds"))
        return _clock_from_seconds(hours * 3600 + minutes * 60 + seconds)
    try:
        return _clock_from_seconds(float(text))
    except ValueError:
        return ""


def _clock_from_seconds(total: float) -> str:
    if total < 0 or total > 7 * 24 * 3600:
        return ""
    whole = int(total)
    hours, remainder = divmod(whole, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _offset_timestamp(start: datetime, clock: str) -> datetime:
    match = re.fullmatch(r"(\d+):(\d{2}):(\d{2})", clock.strip())
    if match is None:
        return start
    hours, minutes, seconds = (int(part) for part in match.groups())
    return start + timedelta(hours=hours, minutes=minutes, seconds=seconds)


def _parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return _parse_epoch(value)
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _parse_epoch(value: Any) -> datetime | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")} or number <= 0:
        return None
    if number > _EPOCH_MS:
        number /= 1000.0
    try:
        return datetime.fromtimestamp(number, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _same_moment(left: datetime | None, right: datetime | None) -> bool:
    if left is None or right is None:
        return False
    if left.tzinfo is None:
        left = left.replace(tzinfo=UTC)
    if right.tzinfo is None:
        right = right.replace(tzinfo=UTC)
    return left.astimezone(UTC) == right.astimezone(UTC)


def _same_optional_moment(left: datetime | None, right: datetime | None) -> bool:
    if left is None and right is None:
        return True
    return _same_moment(left, right)


def _same_duration(left: float | None, right: float | None) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return abs(float(left) - float(right)) < 0.01


def _decode_json(raw: bytes) -> dict[str, Any]:
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise FirefliesApiError("Fireflies returned a response that was not JSON.") from None
    if not isinstance(body, dict):
        raise FirefliesApiError("Fireflies returned an invalid GraphQL response.")
    return body


def _read_error_body(exc: HTTPError) -> bytes:
    try:
        return exc.read()
    except Exception:
        return b""


def _errors_in_body(raw: bytes) -> Any:
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(body, dict):
        return None
    return body.get("errors")


def _handle_http_failure(code: int, headers: Any, raw: bytes, attempt: int) -> None:
    if code == 429:
        wait = _header_retry_after(headers)
        if wait is None:
            wait = _rate_limit_wait(_errors_in_body(raw), None)
        time.sleep(_rate_limit_delay(0.0 if wait is None else wait, attempt))
        return
    if code in _RETRYABLE_HTTP and attempt < 2:
        time.sleep(min(2**attempt, _MAX_RATE_LIMIT_WAIT))
        return
    raise FirefliesApiError(f"Fireflies API request failed with HTTP {code}.")


def _rate_limit_delay(wait: float, attempt: int) -> float:
    """Seconds to sleep. Long waits and exhausted retries raise instead of looping."""
    if wait > _MAX_RATE_LIMIT_WAIT:
        raise FirefliesApiError(
            f"Fireflies API rate limit exceeded. Retry after {max(int(wait), 1)} seconds."
        )
    if attempt >= 5:
        raise FirefliesApiError("Fireflies API rate limit exceeded.")
    return min(max(wait, float(2**attempt)), _MAX_RATE_LIMIT_WAIT)


def _rate_limit_wait(errors: Any, headers: Any) -> float | None:
    """Seconds requested by a GraphQL rate-limit error, or None when it is not one."""
    if not isinstance(errors, list) or not any(
        isinstance(error, dict) and _is_rate_limit_error(error) for error in errors
    ):
        return None
    header_wait = _header_retry_after(headers)
    if header_wait is not None:
        return header_wait
    for error in errors:
        if not isinstance(error, dict) or not _is_rate_limit_error(error):
            continue
        extensions = error.get("extensions") if isinstance(error.get("extensions"), dict) else {}
        metadata = extensions.get("metadata") if isinstance(extensions.get("metadata"), dict) else {}
        parsed = _coerce_retry_after(metadata.get("retryAfter"))
        if parsed is not None:
            return parsed
    return 0.0


def _is_rate_limit_error(error: dict[str, Any]) -> bool:
    extensions = error.get("extensions") if isinstance(error.get("extensions"), dict) else {}
    code = str(extensions.get("code") or "").strip().casefold()
    if code in _RATE_LIMIT_CODES:
        return True
    message = str(error.get("message") or "").casefold()
    return "too many requests" in message or "rate limit" in message


def _header_retry_after(headers: Any) -> float | None:
    if headers is None:
        return None
    raw = headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(float(raw), 0.0)
    except (TypeError, ValueError):
        return None


def _coerce_retry_after(value: Any) -> float | None:
    """Parse a retry hint. Large numbers are epoch milliseconds; smaller ones are seconds."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    if number > _EPOCH_MS:
        return max(number / 1000.0 - time.time(), 0.0)
    if number < 0:
        return None
    return number


if __name__ == "__main__":
    main()
