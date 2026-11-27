"""The sweep (SPEC.md section 4): the expiry pass and the recovery pass, run directly and
through the app's startup pass. Crashes are simulated with SimulatedCrash, which no handler
in the app catches, so the database is left exactly as a dead process would leave it."""

from __future__ import annotations

import asyncio
import time
from datetime import timedelta
from typing import Any
from uuid import UUID

import psycopg
import pytest
from fastapi.testclient import TestClient

from tests.conftest import (
    AVERY,
    FakeAnthropic,
    FakeDiscordRest,
    SimulatedCrash,
    classifier_reply,
    insert_test_session,
    text_reply,
    tool_call,
    tool_turn,
)
from tests.flows import (
    ASK_CHANNEL,
    asker_replies,
    last_user_results,
    make_deps,
    pause,
    receipts,
    request_row,
    seqs,
    session_status,
    ticket,
    tool_results,
)
from triage_agent.agent.pipeline import AskJob, handle_ask
from triage_agent.agent.reply import TokenStore
from triage_agent.agent.resume import handle_click
from triage_agent.app import create_app
from triage_agent.config import ApprovalConfig, Settings
from triage_agent.db.approvals import decide_request
from triage_agent.sweep import STALE_AFTER, SWEEP_INTERVAL_S, run_sweep_once
from triage_agent.tools.close_ticket import execute_close

Db = psycopg.Connection[tuple[Any, ...]]


def expire_now(db: Db, request_id: UUID) -> None:
    db.execute(
        "UPDATE approval_requests SET expires_at = now() - interval '1 minute' WHERE id = %s",
        (request_id,),
    )


def make_stale(db: Db, session_id: UUID) -> None:
    db.execute(
        "UPDATE sessions SET heartbeat_at = now() - interval '11 minutes' WHERE id = %s",
        (session_id,),
    )


def user_result_messages(db: Db, session_id: UUID) -> int:
    rows = db.execute(
        "SELECT content FROM messages WHERE session_id = %s AND role = 'user'", (session_id,)
    ).fetchall()
    return sum(1 for (content,) in rows if tool_results(content))


def test_sweep_cadence_and_staleness_threshold() -> None:
    assert SWEEP_INTERVAL_S == 300
    assert STALE_AFTER == timedelta(minutes=10)


# --- expiry ---------------------------------------------------------------------


@pytest.mark.anyio
async def test_sweep_expires_a_pending_request_and_resumes(
    seeded_database_url: str,
    db: Db,
    fake_anthropic: FakeAnthropic,
    fake_discord: FakeDiscordRest,
    approvals: ApprovalConfig,
) -> None:
    paused = await pause(
        seeded_database_url,
        db,
        fake_anthropic,
        fake_discord,
        approvals,
        tool_call("close_ticket", {"ticket_id": "T-1002"}, "tu_close"),
    )
    expire_now(db, paused.request())
    fake_anthropic.script = [text_reply("The close request expired.")]

    await run_sweep_once(paused.deps)

    req = request_row(db, paused.request())
    assert (req["status"], req["decided_by"], req["self_approved"]) == ("expired", None, False)
    assert req["decided_at"] is not None
    ((_, message_id, content, components),) = fake_discord.channel_edits
    assert message_id == req["discord_message_id"]
    assert "Expired" in content and "UTC" in content and components == []
    (result,) = last_user_results(fake_anthropic)
    assert result == {
        "type": "tool_result",
        "tool_use_id": "tu_close",
        "content": "approval expired",
        "is_error": True,
    }
    assert ticket(db, "T-1002")[0] == "open"
    assert asker_replies(fake_discord) == ["The close request expired."]
    assert session_status(db, paused.session_id)[0] == "completed"


@pytest.mark.anyio
async def test_sweep_leaves_unexpired_requests_alone(
    seeded_database_url: str,
    db: Db,
    fake_anthropic: FakeAnthropic,
    fake_discord: FakeDiscordRest,
    approvals: ApprovalConfig,
) -> None:
    paused = await pause(
        seeded_database_url,
        db,
        fake_anthropic,
        fake_discord,
        approvals,
        tool_call("close_ticket", {"ticket_id": "T-1002"}, "tu_close"),
    )
    await run_sweep_once(paused.deps)
    assert request_row(db, paused.request())["status"] == "pending"
    assert fake_discord.channel_edits == []


@pytest.mark.anyio
async def test_startup_pass_expires_requests_that_lapsed_while_the_app_was_down(
    seeded_database_url: str,
    db: Db,
    fake_anthropic: FakeAnthropic,
    fake_discord: FakeDiscordRest,
    approvals: ApprovalConfig,
    settings: Settings,
) -> None:
    paused = await pause(
        seeded_database_url,
        db,
        fake_anthropic,
        fake_discord,
        approvals,
        tool_call("close_ticket", {"ticket_id": "T-1002"}, "tu_close"),
    )
    expire_now(db, paused.request())
    fake_anthropic.script = [text_reply("It expired.")]

    app = create_app(settings=settings, discord=fake_discord, model=fake_anthropic)
    with TestClient(app):
        deadline = time.monotonic() + 10
        while session_status(db, paused.session_id)[0] != "completed":
            assert time.monotonic() < deadline, "the startup sweep did not finish the session"
            time.sleep(0.05)

    assert request_row(db, paused.request())["status"] == "expired"
    # A restarted app holds no interaction token, so the reply goes out by bot-token REST.
    assert [p.content for p in fake_discord.posts_in(ASK_CHANNEL)] == ["It expired."]


@pytest.mark.anyio
async def test_a_click_after_expires_at_expires_instead_of_approving(
    seeded_database_url: str,
    db: Db,
    fake_anthropic: FakeAnthropic,
    fake_discord: FakeDiscordRest,
    approvals: ApprovalConfig,
) -> None:
    paused = await pause(
        seeded_database_url,
        db,
        fake_anthropic,
        fake_discord,
        approvals,
        tool_call("close_ticket", {"ticket_id": "T-1002"}, "tu_close"),
    )
    expire_now(db, paused.request())
    fake_anthropic.script = [text_reply("Expired.")]

    await handle_click(paused.deps, paused.request(), "approve", AVERY)

    assert request_row(db, paused.request())["status"] == "expired"
    assert ticket(db, "T-1002")[0] == "open"
    assert receipts(db, paused.session_id) == []


@pytest.mark.anyio
async def test_expiry_racing_a_click_gives_one_decision(
    seeded_database_url: str,
    db: Db,
    fake_anthropic: FakeAnthropic,
    fake_discord: FakeDiscordRest,
    approvals: ApprovalConfig,
) -> None:
    paused = await pause(
        seeded_database_url,
        db,
        fake_anthropic,
        fake_discord,
        approvals,
        tool_call("close_ticket", {"ticket_id": "T-1002"}, "tu_close"),
    )
    expire_now(db, paused.request())
    fake_anthropic.script = [text_reply("Done.")]

    await asyncio.gather(
        handle_click(paused.deps, paused.request(), "approve", AVERY),
        run_sweep_once(paused.deps),
    )

    assert len(fake_discord.channel_edits) == 1
    assert asker_replies(fake_discord) == ["Done."]
    assert fake_anthropic.script == []
    assert session_status(db, paused.session_id)[0] == "completed"


# --- recovery ---------------------------------------------------------------------


@pytest.mark.anyio
async def test_recovery_after_the_receipt_but_before_the_results_message(
    seeded_database_url: str,
    db: Db,
    fake_anthropic: FakeAnthropic,
    fake_discord: FakeDiscordRest,
    approvals: ApprovalConfig,
) -> None:
    paused = await pause(
        seeded_database_url,
        db,
        fake_anthropic,
        fake_discord,
        approvals,
        tool_call("close_ticket", {"ticket_id": "T-1002"}, "tu_close"),
    )
    _, notes_before = ticket(db, "T-1002")
    # The click's transaction committed and the close ran, then the process died.
    async with await psycopg.AsyncConnection.connect(seeded_database_url, autocommit=True) as c:
        decision = await decide_request(c, paused.request(), "approved", AVERY)
        assert decision.resume is True
        await execute_close(c, paused.request())
    assert seqs(db, paused.session_id) == [0, 1]
    make_stale(db, paused.session_id)

    model = FakeAnthropic(script=[text_reply("T-1002 is closed.")])
    await run_sweep_once(make_deps(seeded_database_url, model, fake_discord, approvals))

    assert ticket(db, "T-1002") == ("closed", notes_before + 1)
    assert len(receipts(db, paused.session_id)) == 1
    assert seqs(db, paused.session_id) == [0, 1, 2, 3]
    assert asker_replies(fake_discord) == ["T-1002 is closed."]
    assert session_status(db, paused.session_id)[0] == "completed"


@pytest.mark.anyio
async def test_recovery_after_the_results_message_was_appended(
    seeded_database_url: str,
    db: Db,
    fake_anthropic: FakeAnthropic,
    fake_discord: FakeDiscordRest,
    approvals: ApprovalConfig,
) -> None:
    paused = await pause(
        seeded_database_url,
        db,
        fake_anthropic,
        fake_discord,
        approvals,
        tool_call("close_ticket", {"ticket_id": "T-1002"}, "tu_close"),
    )
    _, notes_before = ticket(db, "T-1002")
    fake_anthropic.script = [SimulatedCrash()]  # dies on the model call after the append
    with pytest.raises(SimulatedCrash):
        await handle_click(paused.deps, paused.request(), "approve", AVERY)
    assert seqs(db, paused.session_id) == [0, 1, 2]
    assert session_status(db, paused.session_id)[0] == "running"
    make_stale(db, paused.session_id)

    model = FakeAnthropic(script=[text_reply("T-1002 is closed.")])
    await run_sweep_once(make_deps(seeded_database_url, model, fake_discord, approvals))

    assert seqs(db, paused.session_id) == [0, 1, 2, 3]
    assert user_result_messages(db, paused.session_id) == 1
    assert ticket(db, "T-1002") == ("closed", notes_before + 1)
    assert len(receipts(db, paused.session_id)) == 1
    assert asker_replies(fake_discord) == ["T-1002 is closed."]
    assert session_status(db, paused.session_id)[0] == "completed"


async def crash_during_allowed_turn(
    database_url: str, db: Db, discord: FakeDiscordRest, approvals: ApprovalConfig
) -> UUID:
    session_id = insert_test_session(db, "status of T-1002?", channel=ASK_CHANNEL)
    model = FakeAnthropic(
        script=[
            classifier_reply([("status of T-1002", "other")]),
            tool_turn(tool_call("get_ticket", {"ticket_id": "T-1002"}, "tu_get")),
            SimulatedCrash(),
        ]
    )
    with pytest.raises(SimulatedCrash):
        await handle_ask(
            database_url,
            model,
            discord,
            AskJob(
                session_id=session_id, question="status of T-1002?", application_id="a", token="t"
            ),
            tokens=TokenStore(),
            approvals=approvals,
        )
    assert session_status(db, session_id)[0] == "running"
    return session_id


@pytest.mark.anyio
async def test_recovery_fails_a_crashed_turn_with_only_allowed_calls(
    seeded_database_url: str,
    db: Db,
    fake_discord: FakeDiscordRest,
    approvals: ApprovalConfig,
) -> None:
    sid = await crash_during_allowed_turn(seeded_database_url, db, fake_discord, approvals)
    make_stale(db, sid)
    model = FakeAnthropic()

    await run_sweep_once(make_deps(seeded_database_url, model, fake_discord, approvals))

    assert session_status(db, sid)[0] == "failed"
    assert model.requests == []
    (apology,) = asker_replies(fake_discord)
    assert str(sid) in apology
    assert "Nothing was changed" in apology


@pytest.mark.anyio
async def test_two_sweeps_at_once_recover_a_session_once(
    seeded_database_url: str,
    db: Db,
    fake_discord: FakeDiscordRest,
    approvals: ApprovalConfig,
) -> None:
    sid = await crash_during_allowed_turn(seeded_database_url, db, fake_discord, approvals)
    make_stale(db, sid)
    deps = make_deps(seeded_database_url, FakeAnthropic(), fake_discord, approvals)

    await asyncio.gather(run_sweep_once(deps), run_sweep_once(deps))

    assert len(asker_replies(fake_discord)) == 1


@pytest.mark.anyio
async def test_apology_lists_a_ticket_that_was_already_closed(
    seeded_database_url: str,
    db: Db,
    fake_anthropic: FakeAnthropic,
    fake_discord: FakeDiscordRest,
    approvals: ApprovalConfig,
) -> None:
    paused = await pause(
        seeded_database_url,
        db,
        fake_anthropic,
        fake_discord,
        approvals,
        tool_call("close_ticket", {"ticket_id": "T-1002"}, "tu_close"),
    )
    # The resume closes the ticket, takes one more (read-only) turn, then the process dies.
    fake_anthropic.script = [
        tool_turn(tool_call("get_ticket", {"ticket_id": "T-1002"}, "tu_check")),
        SimulatedCrash(),
    ]
    with pytest.raises(SimulatedCrash):
        await handle_click(paused.deps, paused.request(), "approve", AVERY)
    make_stale(db, paused.session_id)

    model = FakeAnthropic()
    await run_sweep_once(make_deps(seeded_database_url, model, fake_discord, approvals))

    assert model.requests == []
    assert session_status(db, paused.session_id)[0] == "failed"
    (apology,) = asker_replies(fake_discord)
    assert "T-1002 was closed" in apology
    assert str(paused.session_id) in apology
    assert ticket(db, "T-1002")[0] == "closed"
    assert len(receipts(db, paused.session_id)) == 1


@pytest.mark.anyio
async def test_recovery_leaves_live_and_waiting_sessions_alone(
    seeded_database_url: str,
    db: Db,
    fake_anthropic: FakeAnthropic,
    fake_discord: FakeDiscordRest,
    approvals: ApprovalConfig,
) -> None:
    live = await crash_during_allowed_turn(seeded_database_url, db, fake_discord, approvals)
    db.execute(
        "UPDATE sessions SET heartbeat_at = now() - interval '9 minutes' WHERE id = %s", (live,)
    )
    waiting = await pause(
        seeded_database_url,
        db,
        fake_anthropic,
        fake_discord,
        approvals,
        tool_call("close_ticket", {"ticket_id": "T-1002"}, "tu_close"),
    )
    make_stale(db, waiting.session_id)  # old heartbeat, but awaiting approval, not running
    heartbeats_before = db.execute("SELECT id, heartbeat_at FROM sessions ORDER BY id").fetchall()
    replies_before = len(asker_replies(fake_discord))
    model = FakeAnthropic()

    await run_sweep_once(make_deps(seeded_database_url, model, fake_discord, approvals))

    assert session_status(db, live)[0] == "running"
    assert session_status(db, waiting.session_id)[0] == "awaiting_approval"
    assert db.execute("SELECT id, heartbeat_at FROM sessions ORDER BY id").fetchall() == (
        heartbeats_before
    )
    assert len(asker_replies(fake_discord)) == replies_before
    assert model.requests == []
