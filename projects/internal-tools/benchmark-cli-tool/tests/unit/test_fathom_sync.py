"""Offline tests for Fathom meeting ingestion."""

from __future__ import annotations

import inspect
import io
import json
from dataclasses import replace
from datetime import UTC, datetime
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

import pytest
from pydantic import ValidationError

from benchmark_cli.fathom_sync import (
    FathomApiError,
    FathomMeeting,
    FathomRestClient,
    fathom_source_file,
    meeting_digest,
    parse_meeting,
    parse_summary,
    parse_transcript,
    sync_meetings,
    units_for_meeting,
)
from benchmark_cli.ingestion.chunk import chunk_units
from benchmark_cli.ingestion.pipeline import embedding_prefix, metadata_for
from benchmark_cli.models import ChunkMetadata
from benchmark_cli.storage.postgres import FathomSyncState, initialize_database

_MEETING = {
    "title": "Weekly Google Meet",
    "meeting_title": "Weekly sync",
    "recording_id": 123456789,
    "url": "https://fathom.video/calls/123456789",
    "share_url": "https://fathom.video/share/abc",
    "meeting_url": "https://meet.google.com/abc-defg-hij",
    "created_at": "2026-03-01T17:01:30Z",
    "scheduled_start_time": "2026-03-01T16:00:00Z",
    "scheduled_end_time": "2026-03-01T17:00:00Z",
    "recording_start_time": "2026-03-01T16:01:12Z",
    "recording_end_time": "2026-03-01T17:00:55Z",
    "calendar_invitees": [
        {"name": "Alice Johnson", "email": "alice@bmtech.example", "is_external": False},
        {"name": None, "email": "bob@client.example", "is_external": True},
    ],
    "recorded_by": {
        "name": "Alice Johnson",
        "email": "alice@bmtech.example",
        "team": "Engineering",
    },
}
_TRANSCRIPT = {
    "transcript": [
        {
            "speaker": {
                "display_name": "Alice Johnson",
                "matched_calendar_invitee_email": "alice@bmtech.example",
            },
            "text": "Ship the portal. password=hunter2",
            "timestamp": "00:05:32",
        },
        {
            "speaker": {"display_name": "Bob"},
            "text": "Agreed.",
            "timestamp": "00:06:10",
        },
    ]
}
_SUMMARY = "We will ship the portal."


def _response(payload: object) -> object:
    class _Body:
        def read(self) -> bytes:
            return json.dumps(payload).encode()

        def __enter__(self) -> _Body:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    return _Body()


def _meeting(recording_id: int, title: str) -> dict[str, object]:
    payload = json.loads(json.dumps(_MEETING))
    payload["recording_id"] = recording_id
    payload["title"] = title
    payload["url"] = f"https://fathom.video/calls/{recording_id}"
    return payload


def _state(
    meeting: FathomMeeting,
    *,
    content_sha256: str = "existing",
    revised_at: datetime | None = None,
) -> FathomSyncState:
    return FathomSyncState(
        recording_id=meeting.recording_id,
        source_file=fathom_source_file(meeting.recording_id),
        content_sha256=content_sha256,
        title=meeting.title,
        fathom_url=meeting.url,
        join_url=meeting.meeting_url,
        attendees=meeting.attendees,
        recorded_at=meeting.started_at,
        revised_at=meeting.revision if revised_at is None else revised_at,
    )


class _FakeClient:
    def __init__(
        self,
        meetings: list[object],
        transcripts: dict[str, tuple[object, ...]] | None = None,
        summaries: dict[str, str] | None = None,
    ) -> None:
        self._meetings = meetings
        self._transcripts = transcripts or {}
        self._summaries = summaries or {}
        self.transcript_calls: list[str] = []
        self.summary_calls: list[str] = []

    def list_meetings(self) -> list[object]:
        return self._meetings

    def transcript(self, recording_id: str) -> tuple[object, ...]:
        self.transcript_calls.append(recording_id)
        return self._transcripts.get(recording_id, ())

    def summary(self, recording_id: str) -> str:
        self.summary_calls.append(recording_id)
        return self._summaries.get(recording_id, "")


@pytest.mark.unit
def test_meeting_units_redact_secrets_and_keep_fathom_provenance() -> None:
    meeting = parse_meeting(_MEETING)
    assert meeting is not None
    assert parse_meeting({"title": "No id"}) is None
    missing_url = json.loads(json.dumps(_MEETING))
    missing_url["url"] = ""
    missing_url["share_url"] = None
    fallback = parse_meeting(missing_url)
    assert fallback is not None
    assert fallback.url == "https://fathom.video/calls/123456789"
    loaded = replace(
        meeting,
        transcript=parse_transcript(_TRANSCRIPT),
        summary=parse_summary({"summary": {"markdown_formatted": _SUMMARY}}),
    )
    assert fathom_source_file(loaded.recording_id) == "fathom/123456789.md"
    units = units_for_meeting(loaded)
    text = "\n".join(unit.text for unit in units)
    assert "hunter2" not in text
    assert "[REDACTED]" in text
    assert "Alice Johnson (00:05:32):" in text
    assert "https://meet.google.com/abc-defg-hij" in text
    assert "Summary:" in text
    assert "Attendees: Alice Johnson, bob@client.example" in text
    alice = next(unit for unit in units if unit.speaker == "Alice Johnson")
    assert alice.timestamp == datetime(2026, 3, 1, 16, 6, 44, tzinfo=UTC)

    chunks = chunk_units(units, target_tokens=400, max_tokens=500, min_tokens=1)
    metadata = metadata_for(chunks[0], Path("fathom/123456789.md"), "transcript")
    payload = metadata.as_metadata()
    assert payload["source_type"] == "transcript"
    assert payload["kind"] == "meeting_transcript"
    assert payload["title"] == "Weekly Google Meet"
    assert payload["meeting"] == "Weekly Google Meet"
    assert payload["url"] == "https://fathom.video/calls/123456789"
    assert payload["recording_id"] == "123456789"
    assert payload["attendees"] == ["Alice Johnson", "bob@client.example"]
    assert payload["date"] == "2026-03-01"
    assert "Meeting transcript" in embedding_prefix(metadata, "123456789.md")
    assert "https://fathom.video/calls/123456789" in embedding_prefix(metadata, "123456789.md")

    spoken = next(chunk for chunk in chunks if any(unit.speaker for unit in chunk.units))
    speakers = metadata_for(spoken, Path("fathom/123456789.md"), "transcript").speakers
    assert speakers == ["Alice Johnson", "Bob"]


@pytest.mark.unit
def test_plain_transcript_prefix_stays_stable() -> None:
    metadata = ChunkMetadata(
        source_type="transcript",
        date="2026-03-01",
        ingested_at="2026-03-01T00:00:00+00:00",
        meeting="standup",
        section="general",
    )
    assert embedding_prefix(metadata, "standup.txt") == (
        "Meeting transcript: standup; date: 2026-03-01; section: general."
    )


@pytest.mark.unit
def test_meeting_transcript_metadata_requires_provenance() -> None:
    with pytest.raises(ValidationError):
        ChunkMetadata(
            source_type="transcript",
            date="2026-03-01",
            ingested_at="2026-03-01T00:00:00+00:00",
            meeting="Weekly Google Meet",
            kind="meeting_transcript",
        )


@pytest.mark.unit
def test_list_pages_and_sends_the_api_key_header(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = [
        {"limit": 1, "next_cursor": "cursor-2", "items": [_meeting(111, "First"), "skip-me"]},
        {"limit": 1, "next_cursor": None, "items": [_meeting(222, "Second"), {"title": "no id"}]},
    ]
    seen: list[tuple[str, str | None]] = []

    def fake_open(request: object, timeout: int = 30) -> object:
        del timeout
        url = request.full_url  # type: ignore[attr-defined]
        key = request.get_header("X-api-key")  # type: ignore[attr-defined]
        seen.append((url, key))
        return _response(pages[len(seen) - 1])

    monkeypatch.setattr("benchmark_cli.fathom_sync.urlopen", fake_open)
    meetings = FathomRestClient("SECRETKEY").list_meetings()

    assert [meeting.recording_id for meeting in meetings] == ["111", "222"]
    assert all(key == "SECRETKEY" for _, key in seen)
    assert all("SECRETKEY" not in url for url, _ in seen)
    assert "cursor=cursor-2" in seen[1][0]
    assert "/meetings" in seen[0][0]
    assert "include_transcript" not in seen[0][0]


@pytest.mark.unit
def test_missing_transcript_is_empty_and_errors_hide_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(request: object, timeout: int = 30) -> object:
        del request, timeout
        raise HTTPError(
            "https://api.fathom.ai/external/v1/recordings/123/transcript",
            404,
            "missing",
            None,
            io.BytesIO(b"{}"),
        )

    monkeypatch.setattr("benchmark_cli.fathom_sync.urlopen", missing)
    client = FathomRestClient("SECRETKEY")
    assert client.transcript("123456789") == ()
    assert client.summary("123456789") == ""

    def denied(request: object, timeout: int = 30) -> object:
        del request, timeout
        raise HTTPError(
            "https://api.fathom.ai/external/v1/meetings?api_key=SECRETKEY",
            401,
            "denied",
            None,
            io.BytesIO(b'{"api_key":"SECRETKEY"}'),
        )

    monkeypatch.setattr("benchmark_cli.fathom_sync.urlopen", denied)
    with pytest.raises(Exception) as exc_info:
        client.list_meetings()
    message = str(exc_info.value)
    assert "SECRETKEY" not in message
    assert exc_info.value.__cause__ is None


@pytest.mark.unit
def test_rate_limit_retries_without_sleeping_in_the_test(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}

    def limited(request: object, timeout: int = 30) -> object:
        del request, timeout
        attempts["count"] += 1
        if attempts["count"] == 1:
            headers = Message()
            headers["Retry-After"] = "0"
            raise HTTPError(
                "https://api.fathom.ai/external/v1/meetings",
                429,
                "slow",
                headers,
                io.BytesIO(b"{}"),
            )
        return _response({"items": [], "next_cursor": None, "limit": 1})

    monkeypatch.setattr("benchmark_cli.fathom_sync.urlopen", limited)
    monkeypatch.setattr("benchmark_cli.fathom_sync.time.sleep", lambda seconds: None)
    assert FathomRestClient("SECRETKEY").list_meetings() == []
    assert attempts["count"] == 2


@pytest.mark.unit
def test_repeated_cursor_stops_before_a_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    def repeat(request: object, timeout: int = 30) -> object:
        del request, timeout
        return _response({"items": [_meeting(111, "First")], "next_cursor": "again"})

    monkeypatch.setattr("benchmark_cli.fathom_sync.urlopen", repeat)
    with pytest.raises(Exception, match="repeated a cursor"):
        FathomRestClient("key").list_meetings()


@pytest.mark.unit
def test_listing_failure_does_not_delete_stored_meetings(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(source: str) -> None:
        del source
        raise AssertionError("listing failure must not delete stored meetings")

    monkeypatch.setattr("benchmark_cli.fathom_sync.delete_source", fail)

    class Boom:
        def list_meetings(self) -> list[FathomMeeting]:
            raise FathomApiError("Fathom API request failed with HTTP 503.")

    with pytest.raises(FathomApiError, match="503"):
        sync_meetings(Boom())  # type: ignore[arg-type]


@pytest.mark.unit
def test_dry_run_reports_fathom_sources_without_database_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail() -> None:
        raise AssertionError("dry-run must not touch the database")

    monkeypatch.setattr("benchmark_cli.fathom_sync.initialize_database", fail)
    monkeypatch.setattr("benchmark_cli.fathom_sync.delete_source", lambda source: fail())
    monkeypatch.setattr("benchmark_cli.fathom_sync.upsert_fathom_sync_state", lambda state: fail())
    monkeypatch.setattr("benchmark_cli.ingestion.pipeline.initialize_database", fail)

    meeting = parse_meeting(_MEETING)
    assert meeting is not None
    turns = parse_transcript(_TRANSCRIPT)
    client = _FakeClient(
        [meeting],
        transcripts={meeting.recording_id: turns},
        summaries={meeting.recording_id: _SUMMARY},
    )
    reports = sync_meetings(client, dry_run=True)  # type: ignore[arg-type]

    assert len(reports) == 1
    assert reports[0].action == "dry_run"
    assert reports[0].source_file == "fathom/123456789.md"
    assert reports[0].chunks >= 3
    assert client.transcript_calls == ["123456789"]


@pytest.mark.unit
def test_sync_indexes_changed_meetings_and_removes_missing_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unchanged = parse_meeting(_meeting(111, "Unchanged"))
    same_text = parse_meeting(_meeting(222, "Same text"))
    fresh = parse_meeting(_meeting(333, "Fresh"))
    waiting = parse_meeting(_meeting(444, "Waiting"))
    assert unchanged and same_text and fresh and waiting
    turns = parse_transcript(_TRANSCRIPT)
    loaded = replace(same_text, transcript=turns, summary=_SUMMARY)
    digest = meeting_digest(loaded, units_for_meeting(loaded))
    removed = FathomSyncState(
        recording_id="999",
        source_file="fathom/999.md",
        content_sha256="old",
        title="Departed",
        fathom_url="https://fathom.video/calls/999",
        join_url="",
        attendees=(),
        recorded_at=None,
        revised_at=None,
    )
    stored = {
        unchanged.recording_id: _state(unchanged),
        same_text.recording_id: _state(
            same_text,
            content_sha256=digest,
            revised_at=datetime(2020, 1, 1, tzinfo=UTC),
        ),
        removed.recording_id: removed,
    }
    client = _FakeClient(
        [unchanged, same_text, fresh, waiting],
        transcripts={same_text.recording_id: turns, fresh.recording_id: turns},
        summaries={same_text.recording_id: _SUMMARY, fresh.recording_id: _SUMMARY},
    )
    calls: list[object] = []
    monkeypatch.setattr("benchmark_cli.fathom_sync.initialize_database", lambda: calls.append("init"))
    monkeypatch.setattr("benchmark_cli.fathom_sync.fathom_sync_states", lambda: stored)
    monkeypatch.setattr(
        "benchmark_cli.fathom_sync.ingest_units",
        lambda units, **kwargs: calls.append(kwargs) or {"chunks": 4},
    )
    monkeypatch.setattr(
        "benchmark_cli.fathom_sync.delete_source",
        lambda source: calls.append(("delete", source)),
    )
    monkeypatch.setattr(
        "benchmark_cli.fathom_sync.delete_fathom_sync_state",
        lambda recording_id: calls.append(("forget", recording_id)),
    )
    monkeypatch.setattr(
        "benchmark_cli.fathom_sync.upsert_fathom_sync_state",
        lambda row: calls.append(("upsert", row.recording_id)),
    )
    monkeypatch.setattr("benchmark_cli.fathom_sync.create_indexes", lambda: calls.append("indexes"))

    reports = sync_meetings(client, dry_run=False)  # type: ignore[arg-type]
    actions = {report.recording_id: report.action for report in reports}

    assert actions == {
        "111": "unchanged",
        "222": "unchanged",
        "333": "indexed",
        "444": "pending",
        "999": "removed",
    }
    assert client.transcript_calls == ["222", "333", "444"]
    assert client.summary_calls == ["222", "333"]
    indexed = next(item for item in calls if isinstance(item, dict))
    assert indexed["source_type"] == "transcript"
    assert indexed["replace_existing"] is True
    assert indexed["source_filename"] == "fathom/333.md"
    assert ("delete", "fathom/999.md") in calls
    assert ("forget", "999") in calls
    assert ("upsert", "222") in calls
    assert ("upsert", "333") in calls
    assert ("upsert", "111") not in calls
    assert calls[-1] == "indexes"
    fresh_report = next(report for report in reports if report.recording_id == "333")
    assert fresh_report.source_file == "fathom/333.md"
    assert fresh_report.chunks == 4


@pytest.mark.unit
def test_sync_schema_and_workflow_are_wired() -> None:
    package = Path(__file__).resolve().parents[2]
    sql = (package / "src" / "benchmark_cli" / "storage" / "sql" / "006_fathom_sync.sql").read_text(
        encoding="utf-8"
    )
    assert "CREATE TABLE IF NOT EXISTS fathom_sync_state" in sql
    assert "content_sha256" in sql
    assert "revised_at" in sql
    assert "006_fathom_sync.sql" in inspect.getsource(initialize_database)
    pyproject = (package / "pyproject.toml").read_text(encoding="utf-8")
    assert 'benchmark-sync-fathom = "benchmark_cli.fathom_sync:main"' in pyproject

    repo = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / ".github" / "workflows").is_dir()
    )
    workflow = (repo / ".github" / "workflows" / "fathom-ingestion.yml").read_text(encoding="utf-8")
    assert "environment: discord-ingestion" in workflow
    assert 'cron: "17 5 * * *"' in workflow
    assert "FATHOM_API_KEY: ${{ secrets.FATHOM_API_KEY }}" in workflow
    assert "DATABASE_URL: ${{ secrets.DATABASE_URL }}" in workflow
    assert "OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}" in workflow
    assert "benchmark-sync-fathom" in workflow
