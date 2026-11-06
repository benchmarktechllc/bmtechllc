"""Synchronize Fathom meeting transcripts into the retrieval corpus.

Lists meetings visible to ``FATHOM_API_KEY``: calls that key recorded, or that
were shared with that user or their team. A meeting is indexed once its
transcript is available. Each meeting is one source,
``fathom/<recording_id>.md``, with ``source_type`` ``transcript`` and ``kind``
``meeting_transcript``. The summary is stored when Fathom returns one. The
source is replaced as a unit, the same way a portal page or Trello card is
replaced.

The list call does not request transcripts. Fathom counts transcript and
summary calls as heavy requests (about 30 a minute, sometimes fewer). A meeting
is skipped when its recording end time — or ``created_at`` when the recording
has no end time — plus its start time, title, Fathom URL, join URL, and
attendees still match the stored row. The list payload has no separate
``updated_at``. When those fields change, the transcript and summary are
fetched, and a content hash skips the embedding call if the text is unchanged.
A transcript edit that does not move those fields waits until the next change.

Meetings that disappear from the accessible set are deleted from
``document_chunks``. A meeting that is still listed but has no transcript yet
is left untouched and retried next run. ``--dry-run`` still calls Fathom and
counts chunks. It does not embed, write, or delete.

``benchmark-sync-fathom`` needs ``FATHOM_API_KEY``, ``DATABASE_URL``, and
``OPENAI_API_KEY``. The ``fathom-ingestion`` workflow runs on the
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
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import FATHOM_API_BASE_URL, FATHOM_API_KEY
from .ingestion.clean import RawUnit, normalize_text
from .ingestion.pipeline import content_hash, ingest_units
from .storage.postgres import (
    FathomSyncState,
    create_indexes,
    delete_fathom_sync_state,
    delete_source,
    fathom_sync_states,
    initialize_database,
    upsert_fathom_sync_state,
)

FATHOM_SYNC_VERSION = "1"
_MAX_PAGES = 1000
_SAFE_ID = re.compile(r"[^A-Za-z0-9_-]+")


class FathomApiError(RuntimeError):
    """A Fathom API request could not be completed."""


@dataclass(frozen=True, slots=True)
class FathomTurn:
    speaker: str
    text: str
    timestamp: str


@dataclass(frozen=True, slots=True)
class FathomMeeting:
    """One meeting from the list endpoint, plus transcript content when loaded."""

    recording_id: str
    title: str
    url: str
    meeting_url: str
    recorded_by: str
    attendees: tuple[str, ...]
    created_at: datetime | None
    scheduled_start: datetime | None
    recording_start: datetime | None
    recording_end: datetime | None
    transcript: tuple[FathomTurn, ...] = ()
    summary: str = ""

    @property
    def revision(self) -> datetime | None:
        """Skip key. Fathom's list payload has no updated_at."""
        return self.recording_end or self.created_at

    @property
    def started_at(self) -> datetime | None:
        return self.recording_start or self.scheduled_start or self.created_at or self.recording_end


@dataclass(frozen=True, slots=True)
class MeetingSyncReport:
    recording_id: str
    title: str
    source_file: str
    action: str
    chunks: int


class FathomRestClient:
    """REST client for the meeting list and per-recording transcript and summary."""

    def __init__(self, api_key: str, *, api_base_url: str = FATHOM_API_BASE_URL) -> None:
        self._api_key = api_key
        self._api_base_url = api_base_url.rstrip("/")

    def _request_json(
        self,
        path: str,
        params: dict[str, str] | None = None,
        *,
        allow_missing: bool = False,
    ) -> Any:
        url = f"{self._api_base_url}{path}"
        if params:
            url = f"{url}?{urlencode(params)}"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "benchmark-cli-tool fathom-ingestion",
                "X-Api-Key": self._api_key,
            },
        )
        for attempt in range(6):
            try:
                with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed Fathom API URL
                    try:
                        return json.loads(response.read().decode("utf-8"))
                    except json.JSONDecodeError:
                        raise FathomApiError(
                            "Fathom returned a response that was not JSON."
                        ) from None
            except HTTPError as exc:
                code = exc.code
                retry_after = _retry_after(exc)
                exc.close()
                if allow_missing and code == 404:
                    return None
                retryable = code == 429 or code in {500, 502, 503, 504}
                exhausted = (code == 429 and attempt == 5) or (code != 429 and attempt >= 2)
                if not retryable or exhausted:
                    # Drop the cause so a URL or body echoed by the client cannot carry the key.
                    raise FathomApiError(f"Fathom API request failed with HTTP {code}.") from None
                time.sleep(min(max(retry_after, 2**attempt), 60))
            except URLError:
                if attempt >= 2:
                    raise FathomApiError("Fathom API request could not be reached.") from None
                time.sleep(min(2**attempt, 60))
        raise AssertionError("unreachable")

    def list_meetings(self) -> list[FathomMeeting]:
        """Return every meeting the key can see. Transcripts are not included."""
        cursor: str | None = None
        seen_cursors: set[str] = set()
        by_id: dict[str, FathomMeeting] = {}
        order: list[str] = []
        for _page in range(_MAX_PAGES):
            params = {"cursor": cursor} if cursor else None
            payload = self._request_json("/meetings", params)
            if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
                raise FathomApiError("Fathom returned an invalid meetings response.")
            for item in payload["items"]:
                if not isinstance(item, dict):
                    continue
                meeting = parse_meeting(item)
                if meeting is None:
                    continue
                if meeting.recording_id not in by_id:
                    order.append(meeting.recording_id)
                by_id[meeting.recording_id] = meeting
            next_cursor = payload.get("next_cursor")
            if next_cursor is None or not str(next_cursor).strip():
                return [by_id[recording_id] for recording_id in order]
            next_cursor = str(next_cursor).strip()
            if next_cursor in seen_cursors:
                raise FathomApiError("Fathom meeting pagination repeated a cursor.")
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        raise FathomApiError("Fathom meeting pagination did not finish.")

    def transcript(self, recording_id: str) -> tuple[FathomTurn, ...]:
        """Return finalized turns. A missing transcript is empty, not an error."""
        payload = self._request_json(
            f"{_recording_path(recording_id)}/transcript", allow_missing=True
        )
        if payload is None:
            return ()
        return parse_transcript(payload)

    def summary(self, recording_id: str) -> str:
        """Return summary markdown. A missing summary is empty, not an error."""
        payload = self._request_json(f"{_recording_path(recording_id)}/summary", allow_missing=True)
        if payload is None:
            return ""
        return parse_summary(payload)


def parse_meeting(item: dict[str, Any]) -> FathomMeeting | None:
    """Normalize one list item. Items without a numeric recording id are skipped."""
    recording_id = _recording_id(item.get("recording_id"))
    if recording_id is None:
        return None
    title = normalize_text(str(item.get("title") or item.get("meeting_title") or ""))
    title = title or "Untitled meeting"
    url = str(item.get("url") or item.get("share_url") or "").strip()
    if not url:
        # Playback links use /calls/<id>. The list payload normally includes url.
        url = f"https://fathom.video/calls/{recording_id}"
    recorded = item.get("recorded_by") if isinstance(item.get("recorded_by"), dict) else {}
    recorded_by = normalize_text(str(recorded.get("name") or recorded.get("email") or ""))
    return FathomMeeting(
        recording_id=recording_id,
        title=title,
        url=url,
        meeting_url=str(item.get("meeting_url") or "").strip(),
        recorded_by=recorded_by,
        attendees=_attendees(item, recorded_by),
        created_at=_parse_time(item.get("created_at")),
        scheduled_start=_parse_time(item.get("scheduled_start_time")),
        recording_start=_parse_time(item.get("recording_start_time")),
        recording_end=_parse_time(item.get("recording_end_time")),
    )


def parse_transcript(payload: Any) -> tuple[FathomTurn, ...]:
    """Normalize a transcript payload. Empty or not-ready transcripts return no turns."""
    if isinstance(payload, dict):
        items = payload.get("transcript")
    elif isinstance(payload, list):
        items = payload
    else:
        raise FathomApiError("Fathom returned an invalid transcript response.")
    if items is None:
        return ()
    if not isinstance(items, list):
        raise FathomApiError("Fathom returned an invalid transcript response.")
    turns: list[FathomTurn] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        speaker_info = item.get("speaker") if isinstance(item.get("speaker"), dict) else {}
        speaker = normalize_text(str(speaker_info.get("display_name") or "Unknown")) or "Unknown"
        text = normalize_text(str(item.get("text") or ""))
        if not text:
            continue
        turns.append(FathomTurn(speaker, text, str(item.get("timestamp") or "").strip()))
    return tuple(turns)


def parse_summary(payload: Any) -> str:
    """Return summary markdown, or an empty string when Fathom has none yet."""
    if not isinstance(payload, dict):
        raise FathomApiError("Fathom returned an invalid summary response.")
    body = payload.get("summary") if "summary" in payload else payload
    if body is None:
        return ""
    if isinstance(body, dict):
        text = body.get("markdown_formatted")
    else:
        text = body
    if not text:
        return ""
    return normalize_text(str(text))


def fathom_source_file(recording_id: str) -> str:
    safe = _SAFE_ID.sub("", recording_id)
    if not safe:
        raise FathomApiError("Fathom recording id is empty.")
    return f"fathom/{safe}.md"


def units_for_meeting(meeting: FathomMeeting) -> list[RawUnit]:
    """Build retrieval units for one meeting. Every unit carries the same provenance."""
    anchor = meeting.started_at or datetime(1970, 1, 1, tzinfo=UTC)
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
    if meeting.started_at is not None:
        lines.append(f"Date: {meeting.started_at.date().isoformat()}")
    if meeting.url:
        lines.append(f"URL: {meeting.url}")
    if meeting.recorded_by:
        lines.append(f"Recorded by: {meeting.recorded_by}")
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


def meeting_digest(meeting: FathomMeeting, units: Sequence[RawUnit]) -> str:
    recorded = meeting.started_at
    lines = [
        FATHOM_SYNC_VERSION,
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
    client: FathomRestClient,
    *,
    dry_run: bool = False,
    batch_size: int = 128,
) -> list[MeetingSyncReport]:
    """Index finalized meetings and delete ones that left the accessible set."""
    meetings = client.list_meetings()
    if dry_run:
        return [_dry_run_report(client, meeting, batch_size) for meeting in meetings]

    initialize_database()
    stored = fathom_sync_states()
    reports: list[MeetingSyncReport] = []
    seen: set[str] = set()
    for meeting in meetings:
        seen.add(meeting.recording_id)
        source = fathom_source_file(meeting.recording_id)
        previous = stored.get(meeting.recording_id)
        if previous is not None and _listing_matches(previous, meeting):
            reports.append(MeetingSyncReport(meeting.recording_id, meeting.title, source, "unchanged", 0))
            continue
        loaded = _load_content(client, meeting)
        if loaded is None:
            reports.append(MeetingSyncReport(meeting.recording_id, meeting.title, source, "pending", 0))
            continue
        units = units_for_meeting(loaded)
        if not units:
            reports.append(MeetingSyncReport(meeting.recording_id, meeting.title, source, "pending", 0))
            continue
        digest = meeting_digest(loaded, units)
        if (
            previous is not None
            and previous.content_sha256 == digest
            and previous.source_file == source
        ):
            _remember(loaded, source, digest)
            reports.append(MeetingSyncReport(meeting.recording_id, meeting.title, source, "unchanged", 0))
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
            MeetingSyncReport(meeting.recording_id, meeting.title, source, "indexed", int(report["chunks"]))
        )
    for recording_id, previous in stored.items():
        if recording_id in seen:
            continue
        delete_source(previous.source_file)
        delete_fathom_sync_state(recording_id)
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
    if not FATHOM_API_KEY:
        raise SystemExit("FATHOM_API_KEY is required.")
    reports = sync_meetings(
        FathomRestClient(FATHOM_API_KEY),
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    )
    print(json.dumps([asdict(report) for report in reports], indent=2))


def _dry_run_report(client: FathomRestClient, meeting: FathomMeeting, batch_size: int) -> MeetingSyncReport:
    source = fathom_source_file(meeting.recording_id)
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


def _load_content(client: FathomRestClient, meeting: FathomMeeting) -> FathomMeeting | None:
    transcript = client.transcript(meeting.recording_id)
    if not transcript:
        return None
    return replace(meeting, transcript=transcript, summary=client.summary(meeting.recording_id))


def _remember(meeting: FathomMeeting, source: str, digest: str) -> None:
    upsert_fathom_sync_state(
        FathomSyncState(
            recording_id=meeting.recording_id,
            source_file=source,
            content_sha256=digest,
            title=meeting.title,
            fathom_url=meeting.url,
            join_url=meeting.meeting_url,
            attendees=meeting.attendees,
            recorded_at=meeting.started_at,
            revised_at=meeting.revision,
        )
    )


def _listing_matches(previous: FathomSyncState, meeting: FathomMeeting) -> bool:
    return (
        previous.source_file == fathom_source_file(meeting.recording_id)
        and previous.title == meeting.title
        and previous.fathom_url == meeting.url
        and previous.join_url == meeting.meeting_url
        and previous.attendees == meeting.attendees
        and _same_optional_moment(previous.recorded_at, meeting.started_at)
        and _same_optional_moment(previous.revised_at, meeting.revision)
    )


def _attendees(item: dict[str, Any], recorded_by: str) -> tuple[str, ...]:
    names: list[str] = []
    invitees = item.get("calendar_invitees")
    if isinstance(invitees, list):
        for person in invitees:
            if not isinstance(person, dict):
                continue
            label = normalize_text(str(person.get("name") or person.get("email") or ""))
            if label:
                names.append(label)
    if recorded_by and recorded_by not in names:
        names.append(recorded_by)
    return tuple(dict.fromkeys(names))


def _recording_id(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    text = str(value).strip() if not isinstance(value, int) else str(value)
    if not re.fullmatch(r"[0-9]+", text):
        return None
    return text


def _recording_path(recording_id: str) -> str:
    if _recording_id(recording_id) != recording_id:
        raise FathomApiError("Fathom recording id is not numeric.")
    return f"/recordings/{recording_id}"


def _offset_timestamp(start: datetime, clock: str) -> datetime:
    match = re.fullmatch(r"(\d+):(\d{2}):(\d{2})", clock.strip())
    if match is None:
        return start
    hours, minutes, seconds = (int(part) for part in match.groups())
    return start + timedelta(hours=hours, minutes=minutes, seconds=seconds)


def _parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


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


def _retry_after(exc: HTTPError) -> float:
    if exc.headers is None:
        return 0.0
    raw = exc.headers.get("Retry-After")
    if not raw:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


if __name__ == "__main__":
    main()
