"""Offline tests for Fireflies meeting ingestion."""

from __future__ import annotations

import inspect
import io
import json
import time
from dataclasses import replace
from datetime import UTC, datetime
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

import pytest

from benchmark_cli.fireflies_sync import (
    FirefliesApiError,
    FirefliesGraphQLClient,
    FirefliesMeeting,
    fireflies_source_file,
    meeting_digest,
    parse_meeting,
    pending_reason,
    sync_meetings,
    units_for_meeting,
)
from benchmark_cli.ingestion.chunk import chunk_units
from benchmark_cli.ingestion.pipeline import embedding_prefix, metadata_for
from benchmark_cli.storage.postgres import FirefliesSyncState, initialize_database

_MEETING = {
    "id": "abc123",
    "title": "Weekly Google Meet",
    "dateString": "2026-03-01T16:01:12.000Z",
    "date": 1_772_384_472_000,
    "duration": 54.5,
    "transcript_url": "https://app.fireflies.ai/view/abc123",
    "meeting_link": "https://meet.google.com/abc-defg-hij",
    "host_email": "alice@bmtech.example",
    "organizer_email": "alice@bmtech.example",
    "user": {"name": "Alice Johnson", "email": "alice@bmtech.example"},
    "participants": ["alice@bmtech.example", "carol@bmtech.example"],
    "meeting_attendees": [
        {"displayName": "Alice Johnson", "email": "alice@bmtech.example", "name": "Alice"},
        {"displayName": None, "email": "bob@client.example", "name": None},
    ],
    "meeting_info": {"silent_meeting": False, "summary_status": "processed"},
    "is_live": False,
    "sentences": [
        {
            "speaker_name": "Alice Johnson",
            "text": "Ship the portal. password=hunter2",
            "start_time": 332.4,
            "end_time": 340,
        },
        {
            "speaker_name": "Bob",
            "text": "Agreed.",
            "start_time": "00:06:10",
            "end_time": "00:06:12",
        },
    ],
    "summary": {
        "overview": "We will ship the portal.",
        "short_summary": "Ship the portal.",
        "action_items": "Alice sends the checklist.",
    },
}
_SUMMARY = "We will ship the portal.\n\nShip the portal.\n\nAction items:\nAlice sends the checklist."


def _response(payload: object) -> object:
    class _Body:
        headers = None

        def read(self) -> bytes:
            return json.dumps(payload).encode()

        def __enter__(self) -> _Body:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    return _Body()


def _listed(recording_id: str, title: str) -> dict[str, object]:
    payload = json.loads(json.dumps(_MEETING))
    payload["id"] = recording_id
    payload["title"] = title
    payload["transcript_url"] = f"https://app.fireflies.ai/view/{recording_id}"
    payload.pop("sentences", None)
    payload.pop("summary", None)
    return payload


def _state(
    meeting: FirefliesMeeting,
    *,
    content_sha256: str = "existing",
    summary_status: str | None = None,
) -> FirefliesSyncState:
    return FirefliesSyncState(
        recording_id=meeting.recording_id,
        source_file=fireflies_source_file(meeting.recording_id),
        content_sha256=content_sha256,
        title=meeting.title,
        fireflies_url=meeting.url,
        join_url=meeting.meeting_url,
        attendees=meeting.attendees,
        recorded_at=meeting.recorded_at,
        duration_minutes=meeting.duration_minutes,
        host=meeting.host,
        summary_status=meeting.summary_status if summary_status is None else summary_status,
    )


class _FakeClient:
    def __init__(
        self,
        meetings: list[object],
        details: dict[str, object | None] | None = None,
    ) -> None:
        self._meetings = meetings
        self._details = details or {}
        self.detail_calls: list[str] = []

    def list_meetings(self) -> list[object]:
        return self._meetings

    def transcript(self, recording_id: str) -> object | None:
        self.detail_calls.append(recording_id)
        return self._details.get(recording_id)


@pytest.mark.unit
def test_parse_meeting_units_source_path_and_provenance() -> None:
    meeting = parse_meeting(_MEETING)
    assert meeting is not None
    assert parse_meeting({"title": "No id"}) is None
    assert parse_meeting({"id": "../etc/passwd", "title": "Bad"}) is None
    missing_url = json.loads(json.dumps(_MEETING))
    missing_url["transcript_url"] = ""
    fallback = parse_meeting(missing_url)
    assert fallback is not None
    assert fallback.url == "https://app.fireflies.ai/view/abc123"
    epoch_only = json.loads(json.dumps(_MEETING))
    epoch_only.pop("dateString")
    from_epoch = parse_meeting(epoch_only)
    assert from_epoch is not None
    assert from_epoch.recorded_at == datetime.fromtimestamp(1_772_384_472_000 / 1000, UTC)

    listed = json.loads(json.dumps(_MEETING))
    listed["summary"] = {
        "overview": "Same text",
        "short_summary": "Same text",
        "action_items": ["Send notes", "Book follow-up"],
    }
    summarized = parse_meeting(listed)
    assert summarized is not None
    assert summarized.summary == "Same text\n\nAction items:\nSend notes\nBook follow-up"

    assert fireflies_source_file(meeting.recording_id) == "fireflies/abc123.md"
    with pytest.raises(FirefliesApiError, match="invalid"):
        fireflies_source_file("../etc")
    units = units_for_meeting(meeting)
    text = "\n".join(unit.text for unit in units)
    assert "hunter2" not in text
    assert "[REDACTED]" in text
    assert "Alice Johnson (00:05:32):" in text
    assert "Bob (00:06:10):" in text
    assert "https://meet.google.com/abc-defg-hij" in text
    assert "Duration: 54.5 min" in text
    assert "Host: Alice Johnson" in text
    assert "Summary:" in text
    assert "Action items:" in text
    assert "Attendees: Alice Johnson, bob@client.example, carol@bmtech.example" in text
    alice = next(unit for unit in units if unit.speaker == "Alice Johnson")
    assert alice.timestamp == datetime(2026, 3, 1, 16, 6, 44, tzinfo=UTC)

    chunks = chunk_units(units, target_tokens=400, max_tokens=500, min_tokens=1)
    metadata = metadata_for(chunks[0], Path("fireflies/abc123.md"), "transcript")
    payload = metadata.as_metadata()
    assert payload["source_type"] == "transcript"
    assert payload["kind"] == "meeting_transcript"
    assert payload["title"] == "Weekly Google Meet"
    assert payload["meeting"] == "Weekly Google Meet"
    assert payload["url"] == "https://app.fireflies.ai/view/abc123"
    assert payload["recording_id"] == "abc123"
    assert payload["attendees"] == [
        "Alice Johnson",
        "bob@client.example",
        "carol@bmtech.example",
    ]
    assert payload["date"] == "2026-03-01"
    assert "Meeting transcript" in embedding_prefix(metadata, "abc123.md")
    assert "https://app.fireflies.ai/view/abc123" in embedding_prefix(metadata, "abc123.md")

    spoken = next(chunk for chunk in chunks if any(unit.speaker for unit in chunk.units))
    speakers = metadata_for(spoken, Path("fireflies/abc123.md"), "transcript").speakers
    assert speakers == ["Alice Johnson", "Bob"]


@pytest.mark.unit
def test_pending_reason_covers_silent_not_ready_and_empty_sentences() -> None:
    meeting = parse_meeting(_MEETING)
    assert meeting is not None
    assert pending_reason(meeting) is None
    assert pending_reason(meeting, sentences_known=True) is None
    assert pending_reason(replace(meeting, silent=True)) == "silent_meeting"
    assert pending_reason(replace(meeting, live=True)) == "not_ready"
    unknown = replace(meeting, transcript=())
    assert pending_reason(unknown) is None
    assert pending_reason(unknown, sentences_known=True) == "no_sentences"


@pytest.mark.unit
def test_meeting_digest_changes_only_when_indexed_text_changes() -> None:
    meeting = parse_meeting(_MEETING)
    assert meeting is not None
    units = units_for_meeting(meeting)
    digest = meeting_digest(meeting, units)
    assert digest == meeting_digest(meeting, units)
    retitled = replace(meeting, summary_status="processing")
    assert meeting_digest(retitled, units_for_meeting(retitled)) == digest
    edited = replace(
        meeting,
        transcript=meeting.transcript[:1],
    )
    assert meeting_digest(edited, units_for_meeting(edited)) != digest


@pytest.mark.unit
def test_list_pages_without_sentences_and_sends_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = [_listed(f"m{index:02d}", f"Meeting {index}") for index in range(50)]
    pages = [
        {"data": {"transcripts": first}},
        {
            "data": {
                "transcripts": [_listed("last", "Last"), {"title": "no id"}, "skip-me"],
            }
        },
    ]
    seen: list[dict[str, object]] = []

    def fake_open(request: object, timeout: int = 30) -> object:
        del timeout
        body = json.loads(request.data.decode())  # type: ignore[attr-defined]
        body["auth"] = request.get_header("Authorization")  # type: ignore[attr-defined]
        body["url"] = request.full_url  # type: ignore[attr-defined]
        seen.append(body)
        return _response(pages[len(seen) - 1])

    monkeypatch.setattr("benchmark_cli.fireflies_sync.urlopen", fake_open)
    meetings = FirefliesGraphQLClient("SECRETKEY").list_meetings()

    assert [meeting.recording_id for meeting in meetings] == [
        *[f"m{index:02d}" for index in range(50)],
        "last",
    ]
    assert all(item["auth"] == "Bearer SECRETKEY" for item in seen)
    assert all("SECRETKEY" not in str(item["url"]) for item in seen)
    assert all("sentences" not in str(item["query"]) for item in seen)
    assert all("action_items" not in str(item["query"]) for item in seen)
    assert seen[0]["variables"] == {"limit": 50, "skip": 0}
    assert seen[1]["variables"] == {"limit": 50, "skip": 50}


@pytest.mark.unit
def test_detail_requests_sentences_for_one_id(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_open(request: object, timeout: int = 30) -> object:
        del timeout
        seen["body"] = json.loads(request.data.decode())  # type: ignore[attr-defined]
        seen["auth"] = request.get_header("Authorization")  # type: ignore[attr-defined]
        return _response({"data": {"transcript": None}})

    monkeypatch.setattr("benchmark_cli.fireflies_sync.urlopen", fake_open)
    client = FirefliesGraphQLClient("SECRETKEY")
    assert client.transcript("abc123") is None
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["variables"] == {"id": "abc123"}
    assert "sentences" in body["query"]
    assert "short_summary" in body["query"]
    assert seen["auth"] == "Bearer SECRETKEY"
    with pytest.raises(FirefliesApiError, match="invalid"):
        client.transcript("../etc")


@pytest.mark.unit
def test_errors_hide_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(request: object, timeout: int = 30) -> object:
        del request, timeout
        raise HTTPError(
            "https://api.fireflies.ai/graphql?api_key=SECRETKEY",
            401,
            "denied",
            None,
            io.BytesIO(b'{"api_key":"SECRETKEY"}'),
        )

    monkeypatch.setattr("benchmark_cli.fireflies_sync.urlopen", denied)
    with pytest.raises(Exception) as exc_info:
        FirefliesGraphQLClient("SECRETKEY").list_meetings()
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
                "https://api.fireflies.ai/graphql",
                429,
                "slow",
                headers,
                io.BytesIO(b"{}"),
            )
        return _response({"data": {"transcripts": []}})

    monkeypatch.setattr("benchmark_cli.fireflies_sync.urlopen", limited)
    monkeypatch.setattr("benchmark_cli.fireflies_sync.time.sleep", lambda seconds: None)
    assert FirefliesGraphQLClient("SECRETKEY").list_meetings() == []
    assert attempts["count"] == 2


@pytest.mark.unit
def test_graphql_rate_limit_uses_retry_after_epoch_ms(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}

    def limited(request: object, timeout: int = 30) -> object:
        del request, timeout
        attempts["count"] += 1
        if attempts["count"] == 1:
            return _response(
                {
                    "errors": [
                        {
                            "message": "Too many requests SECRETKEY",
                            "extensions": {
                                "code": "too_many_requests",
                                "metadata": {"retryAfter": (time.time() + 1) * 1000},
                            },
                        }
                    ]
                }
            )
        return _response({"data": {"transcripts": []}})

    monkeypatch.setattr("benchmark_cli.fireflies_sync.urlopen", limited)
    monkeypatch.setattr("benchmark_cli.fireflies_sync.time.sleep", lambda seconds: None)
    assert FirefliesGraphQLClient("SECRETKEY").list_meetings() == []
    assert attempts["count"] == 2


@pytest.mark.unit
def test_long_retry_after_is_not_retried_early(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}

    def limited(request: object, timeout: int = 30) -> object:
        del request, timeout
        attempts["count"] += 1
        headers = Message()
        headers["Retry-After"] = "3600"
        raise HTTPError(
            "https://api.fireflies.ai/graphql",
            429,
            "daily",
            headers,
            io.BytesIO(b'{"errors":[{"message":"SECRETKEY"}]}'),
        )

    monkeypatch.setattr("benchmark_cli.fireflies_sync.urlopen", limited)

    def fail_sleep(seconds: float) -> None:
        raise AssertionError(f"must not retry before Retry-After ({seconds})")

    monkeypatch.setattr("benchmark_cli.fireflies_sync.time.sleep", fail_sleep)
    with pytest.raises(FirefliesApiError, match="3600") as exc_info:
        FirefliesGraphQLClient("SECRETKEY").list_meetings()
    assert "SECRETKEY" not in str(exc_info.value)
    assert attempts["count"] == 1


@pytest.mark.unit
def test_repeated_page_stops_before_a_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    page = [_listed(f"m{index:02d}", f"Meeting {index}") for index in range(50)]

    def repeat(request: object, timeout: int = 30) -> object:
        del request, timeout
        return _response({"data": {"transcripts": page}})

    monkeypatch.setattr("benchmark_cli.fireflies_sync.urlopen", repeat)
    with pytest.raises(FirefliesApiError, match="repeated a page"):
        FirefliesGraphQLClient("key").list_meetings()


@pytest.mark.unit
def test_listing_failure_does_not_delete_stored_meetings(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(source: str) -> None:
        del source
        raise AssertionError("listing failure must not delete stored meetings")

    monkeypatch.setattr("benchmark_cli.fireflies_sync.delete_source", fail)

    class Boom:
        def list_meetings(self) -> list[FirefliesMeeting]:
            raise FirefliesApiError("Fireflies API request failed with HTTP 503.")

    with pytest.raises(FirefliesApiError, match="503"):
        sync_meetings(Boom())  # type: ignore[arg-type]


@pytest.mark.unit
def test_dry_run_reports_fireflies_sources_without_database_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail() -> None:
        raise AssertionError("dry-run must not touch the database")

    monkeypatch.setattr("benchmark_cli.fireflies_sync.initialize_database", fail)
    monkeypatch.setattr("benchmark_cli.fireflies_sync.delete_source", lambda source: fail())
    monkeypatch.setattr(
        "benchmark_cli.fireflies_sync.upsert_fireflies_sync_state", lambda state: fail()
    )
    monkeypatch.setattr("benchmark_cli.ingestion.pipeline.initialize_database", fail)

    meeting = parse_meeting(_listed("abc123", "Weekly Google Meet"))
    assert meeting is not None
    loaded = parse_meeting(_MEETING)
    assert loaded is not None
    client = _FakeClient([meeting], details={meeting.recording_id: loaded})
    reports = sync_meetings(client, dry_run=True)  # type: ignore[arg-type]

    assert len(reports) == 1
    assert reports[0].action == "dry_run"
    assert reports[0].source_file == "fireflies/abc123.md"
    assert reports[0].chunks >= 3
    assert client.detail_calls == ["abc123"]


@pytest.mark.unit
def test_sync_indexes_changed_meetings_and_removes_missing_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unchanged = parse_meeting(_listed("aaa", "Unchanged"))
    same_text = parse_meeting(_listed("bbb", "Same text"))
    fresh = parse_meeting(_listed("ccc", "Fresh"))
    waiting = parse_meeting(_listed("ddd", "Waiting"))
    silent = parse_meeting(_listed("eee", "Silent"))
    assert unchanged and same_text and fresh and waiting and silent
    fresh = replace(fresh, summary_status="processing")
    silent = replace(silent, silent=True)
    turns = parse_meeting(_MEETING)
    assert turns is not None
    loaded_same = replace(same_text, transcript=turns.transcript, summary=turns.summary)
    digest = meeting_digest(loaded_same, units_for_meeting(loaded_same))
    removed = FirefliesSyncState(
        recording_id="zzz",
        source_file="fireflies/zzz.md",
        content_sha256="old",
        title="Departed",
        fireflies_url="https://app.fireflies.ai/view/zzz",
        join_url="",
        attendees=(),
        recorded_at=None,
        duration_minutes=None,
        host="",
        summary_status="",
    )
    stored = {
        unchanged.recording_id: _state(unchanged),
        same_text.recording_id: _state(
            same_text,
            content_sha256=digest,
            summary_status="processing",
        ),
        removed.recording_id: removed,
    }
    client = _FakeClient(
        [unchanged, same_text, fresh, waiting, silent],
        details={
            same_text.recording_id: loaded_same,
            fresh.recording_id: replace(fresh, transcript=turns.transcript, summary=turns.summary),
            waiting.recording_id: replace(waiting, transcript=(), summary=""),
        },
    )
    calls: list[object] = []
    monkeypatch.setattr(
        "benchmark_cli.fireflies_sync.initialize_database", lambda: calls.append("init")
    )
    monkeypatch.setattr("benchmark_cli.fireflies_sync.fireflies_sync_states", lambda: stored)
    monkeypatch.setattr(
        "benchmark_cli.fireflies_sync.ingest_units",
        lambda units, **kwargs: calls.append(kwargs) or {"chunks": 4},
    )
    monkeypatch.setattr(
        "benchmark_cli.fireflies_sync.delete_source",
        lambda source: calls.append(("delete", source)),
    )
    monkeypatch.setattr(
        "benchmark_cli.fireflies_sync.delete_fireflies_sync_state",
        lambda recording_id: calls.append(("forget", recording_id)),
    )
    monkeypatch.setattr(
        "benchmark_cli.fireflies_sync.upsert_fireflies_sync_state",
        lambda row: calls.append(("upsert", row.recording_id, row.summary_status)),
    )
    monkeypatch.setattr("benchmark_cli.fireflies_sync.create_indexes", lambda: calls.append("indexes"))

    reports = sync_meetings(client, dry_run=False)  # type: ignore[arg-type]
    actions = {report.recording_id: report.action for report in reports}

    assert actions == {
        "aaa": "unchanged",
        "bbb": "unchanged",
        "ccc": "indexed",
        "ddd": "pending",
        "eee": "pending",
        "zzz": "removed",
    }
    assert client.detail_calls == ["bbb", "ccc", "ddd"]
    indexed = next(item for item in calls if isinstance(item, dict))
    assert indexed["source_type"] == "transcript"
    assert indexed["replace_existing"] is True
    assert indexed["source_filename"] == "fireflies/ccc.md"
    assert ("delete", "fireflies/zzz.md") in calls
    assert ("forget", "zzz") in calls
    assert ("delete", "fireflies/eee.md") not in calls
    assert ("upsert", "bbb", "processed") in calls
    assert ("upsert", "ccc", "processing") in calls
    assert ("upsert", "aaa", "processed") not in calls
    assert calls[-1] == "indexes"
    fresh_report = next(report for report in reports if report.recording_id == "ccc")
    assert fresh_report.source_file == "fireflies/ccc.md"
    assert fresh_report.chunks == 4


@pytest.mark.unit
def test_sync_schema_and_workflow_are_wired() -> None:
    package = Path(__file__).resolve().parents[2]
    sql = (
        package / "src" / "benchmark_cli" / "storage" / "sql" / "007_fireflies_sync.sql"
    ).read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS fireflies_sync_state" in sql
    assert "content_sha256" in sql
    assert "summary_status" in sql
    assert "007_fireflies_sync.sql" in inspect.getsource(initialize_database)
    pyproject = (package / "pyproject.toml").read_text(encoding="utf-8")
    assert 'benchmark-sync-fireflies = "benchmark_cli.fireflies_sync:main"' in pyproject
    assert 'benchmark-sync-fathom = "benchmark_cli.fathom_sync:main"' in pyproject

    repo = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / ".github" / "workflows").is_dir()
    )
    workflow = (repo / ".github" / "workflows" / "fireflies-ingestion.yml").read_text(
        encoding="utf-8"
    )
    assert "name: Fireflies ingestion" in workflow
    assert "environment: discord-ingestion" in workflow
    assert 'cron: "47 5 * * *"' in workflow
    assert "FIREFLIES_API_KEY: ${{ secrets.FIREFLIES_API_KEY }}" in workflow
    assert "DATABASE_URL: ${{ secrets.DATABASE_URL }}" in workflow
    assert "OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}" in workflow
    assert "benchmark-sync-fireflies" in workflow
    assert "timeout-minutes: 90" in workflow
    fathom = (repo / ".github" / "workflows" / "fathom-ingestion.yml").read_text(encoding="utf-8")
    assert "benchmark-sync-fathom" in fathom
    assert "FATHOM_API_KEY: ${{ secrets.FATHOM_API_KEY }}" in fathom
