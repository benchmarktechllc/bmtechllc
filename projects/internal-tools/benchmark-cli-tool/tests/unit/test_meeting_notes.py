from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from benchmark_cli import meeting_notes
from benchmark_cli.discord_ask import AskConfig
from benchmark_cli.fireflies_sync import FirefliesMeeting, FirefliesTurn

NOW = datetime(2026, 11, 16, 15, 0, tzinfo=UTC)
URL = "https://app.fireflies.ai/view/abc123"


def meeting(
    recording_id: str = "m1",
    *,
    at: datetime | None = NOW - timedelta(hours=1),
    live: bool = False,
    silent: bool = False,
    sentences: bool = False,
) -> FirefliesMeeting:
    turns = (FirefliesTurn("Ann", "We reviewed the launch.", "00:00:01"),) if sentences else ()
    return FirefliesMeeting(
        recording_id=recording_id,
        title="Sync",
        url=URL,
        meeting_url="",
        host="",
        attendees=(),
        recorded_at=at,
        duration_minutes=10.0,
        summary_status="",
        silent=silent,
        live=live,
        transcript=turns,
    )


class FakeStore:
    def __init__(self, watermark: datetime | None, posted: set[str] | None = None) -> None:
        self.mark = watermark
        self.done = set(posted or ())
        self.records: list[str] = []

    def watermark(self) -> datetime | None:
        return self.mark

    def set_watermark(self, value: datetime) -> None:
        self.mark = value

    def posted(self, transcript_id: str) -> bool:
        return transcript_id in self.done

    def record(self, transcript_id, channel_id, message_ids, title) -> bool:
        if transcript_id in self.done:
            return False
        self.done.add(transcript_id)
        self.records.append(transcript_id)
        return True


class FakeClient:
    def __init__(self, listed: list[FirefliesMeeting], details: dict[str, FirefliesMeeting]):
        self.listed = listed
        self.details = details
        self.list_calls = 0
        self.detail_calls: list[str] = []

    def list_meetings(self) -> list[FirefliesMeeting]:
        self.list_calls += 1
        return self.listed

    def transcript(self, recording_id: str) -> FirefliesMeeting | None:
        self.detail_calls.append(recording_id)
        return self.details.get(recording_id)


def run(client: FakeClient, store: FakeStore):
    sent: list[list[str]] = []

    async def send(parts: list[str]) -> list[str]:
        sent.append(parts)
        return [str(index) for index in range(len(parts))]

    result = asyncio.run(
        meeting_notes.run_tick(
            client,  # type: ignore[arg-type]
            store,
            channel_id="1",
            summarize=lambda loaded: f"summary {loaded.recording_id}",
            send=send,
            now=NOW,
        )
    )
    return result, sent


RAW = """Here is the recap:
**Nov 1**
## Discussion:
- Launch plan reviewed
* Budget
ToDos
1. Send deck (Ann)
https://example.com/x
"""


def test_normalize_formats_date_bullets_and_link() -> None:
    text = meeting_notes.normalize_summary(RAW, meeting(sentences=True))
    lines = text.split("\n")
    assert lines[0] == "**November 16, 2026**"
    assert "Discussion" in lines and "ToDos" in lines
    assert "• Launch plan reviewed" in lines and "• Budget" in lines
    assert "• Send deck (Ann)" in lines
    assert "Here is" not in text and "example.com" not in text
    assert lines[-1] == f"<{URL}>"


def test_date_is_new_york_without_zero_padding() -> None:
    at = datetime(2026, 11, 5, 2, 0, tzinfo=UTC)
    assert meeting_notes.date_title(at) == "November 4, 2026"


def test_owner_is_not_added_when_absent() -> None:
    text = meeting_notes.normalize_summary("Discussion\n• A\nToDos\n• Write doc", meeting())
    assert "• Write doc\n" in text
    assert "(" not in text


def test_trims_discussion_before_todos_and_split_keeps_link_last() -> None:
    discussion = "\n".join(f"• topic {index} " + "x" * 80 for index in range(60))
    raw = f"Discussion\n{discussion}\nToDos\n• Keep this (Ann)"
    text = meeting_notes.normalize_summary(raw, meeting())
    assert len(text) <= meeting_notes.NOTES_LIMIT
    assert "• Keep this (Ann)" in text
    big = meeting_notes.normalize_summary(raw, meeting(), max_chars=10_000)
    parts = meeting_notes.split_notes(big)
    assert len(parts) > 1 and all(len(part) <= 2000 for part in parts)
    assert parts[0].startswith("**November 16, 2026**")
    assert parts[-1].endswith(f"<{URL}>")


def test_first_enable_sets_watermark_and_posts_nothing() -> None:
    client = FakeClient([meeting(sentences=True)], {})
    store = FakeStore(None)
    result, sent = run(client, store)
    assert result.initialized and store.mark == NOW
    assert sent == [] and client.list_calls == 0


def test_posts_ready_meeting_once_and_advances_watermark() -> None:
    ready = meeting("m1", sentences=True)
    store = FakeStore(NOW - timedelta(days=1))
    client = FakeClient([ready], {"m1": ready})
    result, sent = run(client, store)
    assert result.posted == 1 and len(sent) == 1
    assert store.records == ["m1"] and store.mark == NOW
    result, sent = run(client, store)
    assert result.posted == 0 and sent == [] and client.detail_calls == ["m1"]


def test_already_posted_is_skipped_without_detail_fetch() -> None:
    client = FakeClient([meeting("m1", sentences=True)], {})
    result, sent = run(client, FakeStore(NOW - timedelta(days=1), {"m1"}))
    assert sent == [] and client.detail_calls == []


def test_older_than_watermark_is_ignored() -> None:
    old = meeting("old", at=NOW - timedelta(days=3), sentences=True)
    client = FakeClient([old], {"old": old})
    _, sent = run(client, FakeStore(NOW - timedelta(days=1)))
    assert sent == [] and client.detail_calls == []


def test_naive_datetimes_are_utc() -> None:
    naive = meeting("n", at=(NOW - timedelta(hours=1)).replace(tzinfo=None), sentences=True)
    client = FakeClient([naive], {"n": naive})
    result, _ = run(client, FakeStore((NOW - timedelta(days=1)).replace(tzinfo=None)))
    assert result.posted == 1


def test_pending_meeting_holds_watermark() -> None:
    start = NOW - timedelta(days=1)
    live = meeting("live", at=NOW - timedelta(hours=3), live=True)
    ready = meeting("ready", at=NOW - timedelta(hours=1), sentences=True)
    store = FakeStore(start)
    client = FakeClient([ready, live], {"ready": ready})
    result, sent = run(client, store)
    assert result.posted == 1 and result.pending == 1
    assert store.mark == live.recorded_at
    assert client.detail_calls == ["ready"]


def test_no_sentences_holds_and_silent_does_not() -> None:
    start = NOW - timedelta(days=1)
    empty = meeting("empty", at=NOW - timedelta(hours=2))
    quiet = meeting("quiet", at=NOW - timedelta(hours=3), silent=True)
    store = FakeStore(start)
    client = FakeClient([empty, quiet], {"empty": empty, "quiet": quiet})
    result, sent = run(client, store)
    assert sent == [] and store.mark == empty.recorded_at
    assert "quiet" not in client.detail_calls

    store = FakeStore(start)
    result, sent = run(FakeClient([quiet], {}), store)
    assert sent == [] and result.skipped == 1 and store.mark == NOW


def test_failed_send_is_not_recorded_and_holds() -> None:
    ready = meeting("m1", sentences=True)
    store = FakeStore(NOW - timedelta(days=1))
    client = FakeClient([ready], {"m1": ready})

    async def boom(parts: list[str]) -> list[str]:
        raise RuntimeError("discord down")

    result = asyncio.run(
        meeting_notes.run_tick(
            client,  # type: ignore[arg-type]
            store,
            channel_id="1",
            summarize=lambda loaded: "x",
            send=boom,
            now=NOW,
        )
    )
    assert result.posted == 0 and store.records == [] and store.mark == ready.recorded_at


def test_flag_off_does_not_start_or_tick() -> None:
    off = AskConfig(100000000000000001, None)
    assert not meeting_notes.should_start_meeting_notes(off, "key")
    on = AskConfig(100000000000000001, None, True, 100000000000000002, 600)
    assert meeting_notes.should_start_meeting_notes(on, "key")
    assert not meeting_notes.should_start_meeting_notes(on, "")
    no_channel = AskConfig(100000000000000001, None, True, None, 600)
    assert not meeting_notes.should_start_meeting_notes(no_channel, "key")

    poller = meeting_notes.MeetingNotesPoller(
        object(),  # type: ignore[arg-type]
        off,
        FakeClient([], {}),  # type: ignore[arg-type]
        store=FakeStore(None),
    )
    assert asyncio.run(poller.tick()) is None


def test_typographic_hyphens_become_ascii() -> None:
    raw = "Discussion\n• Non‑breaking – en — em\nToDos\n• Ship‑it (Ann)"
    text = meeting_notes.normalize_summary(raw, meeting())
    assert "• Non-breaking - en - em" in text
    assert "• Ship-it (Ann)" in text
    assert not any(c in text for c in "‑–—")


def test_build_messages_excludes_fireflies_action_items() -> None:
    from dataclasses import replace

    base = meeting(sentences=True)
    combined = "Overview text.\n\nAction items:\nBob: fix the thing"
    loaded = replace(base, summary=combined, overview="Overview text.")
    user = meeting_notes.build_messages(loaded)[1]["content"]
    assert "Overview text." in user
    assert "source of truth" in user
    assert "Action items" not in user
    assert "Bob: fix the thing" not in user
    assert "Ann: We reviewed the launch." in user


def test_build_messages_omits_hint_without_overview() -> None:
    from dataclasses import replace

    loaded = replace(meeting(sentences=True), summary="Action items:\nBob: x")
    user = meeting_notes.build_messages(loaded)[1]["content"]
    assert "Bob: x" not in user
    assert "hint" not in user


def test_parse_meeting_overview_excludes_action_items() -> None:
    from benchmark_cli.fireflies_sync import parse_meeting

    item = {
        "id": "abc123",
        "title": "T",
        "summary": {"overview": "", "short_summary": "Short.", "action_items": "Bob: x"},
    }
    parsed = parse_meeting(item)
    assert parsed is not None
    assert parsed.overview == "Short."
    assert "Bob" in parsed.summary


def _meeting_notes_tier(env: dict[str, str]) -> list[str]:
    import json
    import os
    import subprocess
    import sys

    code = (
        "import json; from benchmark_cli.config import get_model_config as g;"
        "m=g('meeting_notes'); s=g('synthesis');"
        "print(json.dumps([m.model, s.model, m.input_cost_per_million, m.output_cost_per_million]))"
    )
    base = {k: v for k, v in os.environ.items() if not k.startswith(("MEETING_NOTES_", "SYNTHESIS_"))}
    out = subprocess.run(
        [sys.executable, "-c", code], env={**base, **env}, capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


def test_meeting_notes_model_defaults_to_synthesis_model() -> None:
    model, synthesis, inp, out = _meeting_notes_tier({})
    assert model == synthesis
    assert (inp, out) == (0.25, 2.0)
    assert _meeting_notes_tier({"MEETING_NOTES_MODEL": ""})[0] == synthesis


def test_meeting_notes_model_env_overrides_without_touching_synthesis() -> None:
    model, synthesis, inp, out = _meeting_notes_tier(
        {
            "MEETING_NOTES_MODEL": "gpt-5",
            "MEETING_NOTES_INPUT_COST_PER_MILLION": "1.25",
            "MEETING_NOTES_OUTPUT_COST_PER_MILLION": "10",
        }
    )
    assert (model, synthesis, inp, out) == ("gpt-5", "gpt-5-mini", 1.25, 10.0)


def _capture_request(monkeypatch, model: str, effort: str) -> dict:  # type: ignore[no-untyped-def]
    from types import SimpleNamespace

    from benchmark_cli import config, llm

    requests: list[dict] = []  # type: ignore[type-arg]

    def create(**kwargs):  # type: ignore[no-untyped-def]
        requests.append(kwargs)
        message = SimpleNamespace(content="**Discussion**\n- a\n**ToDos**\n- b")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    monkeypatch.setattr(llm, "_record", lambda *_a, **_k: None)
    tiers = dict(config.MODEL_TIERS)
    tiers["meeting_notes"] = config.ModelConfig(model, 0.25, 2.0)
    monkeypatch.setattr(config, "MODEL_TIERS", tiers)
    monkeypatch.setattr(config, "MEETING_NOTES_REASONING_EFFORT", effort)
    meeting_notes.generate_summary(meeting(sentences=True))
    return requests[0]


def test_generate_summary_sends_reasoning_effort_only_for_gpt5(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    request = _capture_request(monkeypatch, "gpt-5", "low")
    assert request["model"] == "gpt-5"
    assert request["reasoning_effort"] == "low"
    assert "reasoning_effort" not in _capture_request(monkeypatch, "gpt-4.1", "low")
    assert "reasoning_effort" not in _capture_request(monkeypatch, "gpt-5", "")
