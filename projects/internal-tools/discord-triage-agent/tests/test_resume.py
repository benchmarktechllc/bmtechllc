"""Resume after the last decision of a paused turn (SPEC.md section 5, Resume) and reply
routing (SPEC.md section 2, step 4)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import psycopg
import pytest

from tests.conftest import (
    APPROVER_CHANNEL,
    ASKER,
    AVERY,
    BLAKE,
    FakeAnthropic,
    FakeDiscordRest,
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
    result_json,
    seqs,
    session_status,
    ticket,
    tool_results,
)
from triage_agent.agent.reply import TokenStore
from triage_agent.agent.resume import handle_click, resume_session
from triage_agent.config import ApprovalConfig

Db = psycopg.Connection[tuple[Any, ...]]


def agent_requests(model: FakeAnthropic) -> list[Any]:
    return [r for r in model.requests if r.model == "claude-sonnet-5-5"]


@pytest.mark.anyio
async def test_one_approved_and_one_denied_resume_once_with_every_result_in_order(
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
        tool_call("get_ticket", {"ticket_id": "T-1002"}, "tu_get"),
        tool_call("close_ticket", {"ticket_id": "T-1002"}, "tu_a"),
        tool_call("delete_ticket", {"ticket_id": "T-1003"}, "tu_x"),
        tool_call("close_ticket", {"ticket_id": "T-1005"}, "tu_b"),
    )
    fake_anthropic.script = [text_reply("T-1002 closed; closing T-1005 was denied.")]

    await handle_click(paused.deps, paused.request("tu_a"), "approve", AVERY)
    # One request is still pending: no resume, no model call, no reply yet.
    assert len(agent_requests(fake_anthropic)) == 1
    assert session_status(db, paused.session_id)[0] == "awaiting_approval"
    assert asker_replies(fake_discord) == []

    await handle_click(paused.deps, paused.request("tu_b"), "deny", BLAKE)

    assert len(agent_requests(fake_anthropic)) == 2
    results = last_user_results(fake_anthropic)
    assert [r["tool_use_id"] for r in results] == ["tu_get", "tu_a", "tu_x", "tu_b"]
    assert "Birch & Co" in results[0]["content"] and "is_error" not in results[0]
    assert result_json(results[1]) == {
        "ticket_id": "T-1002",
        "status": "closed",
        "already_closed": False,
    }
    assert results[2]["is_error"] is True
    assert results[3] == {
        "type": "tool_result",
        "tool_use_id": "tu_b",
        "content": "denied by Blake Synthetic",
        "is_error": True,
    }
    assert ticket(db, "T-1002")[0] == "closed"
    assert ticket(db, "T-1005")[0] == "open"
    assert len(receipts(db, paused.session_id)) == 1
    assert asker_replies(fake_discord) == ["T-1002 closed; closing T-1005 was denied."]
    assert session_status(db, paused.session_id)[0] == "completed"


@pytest.mark.anyio
async def test_denied_by_falls_back_to_the_user_id_for_unknown_approvers(
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
    fake_anthropic.script = [text_reply("Denied.")]
    await handle_click(paused.deps, paused.request(), "deny", "1999")
    (result,) = last_user_results(fake_anthropic)
    assert result["content"] == "denied by 1999"


@pytest.mark.anyio
async def test_two_approvals_decided_at_the_same_moment_resume_once(
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
        tool_call("close_ticket", {"ticket_id": "T-1002"}, "tu_a"),
        tool_call("close_ticket", {"ticket_id": "T-1005"}, "tu_b"),
    )
    fake_anthropic.script = [text_reply("Both closed.")]

    await asyncio.gather(
        handle_click(paused.deps, paused.request("tu_a"), "approve", AVERY),
        handle_click(paused.deps, paused.request("tu_b"), "approve", BLAKE),
    )

    assert len(agent_requests(fake_anthropic)) == 2  # the pause, then exactly one resume
    assert [r["tool_use_id"] for r in last_user_results(fake_anthropic)] == ["tu_a", "tu_b"]
    assert asker_replies(fake_discord) == ["Both closed."]
    assert len(receipts(db, paused.session_id)) == 2
    assert fake_anthropic.script == []


@pytest.mark.anyio
async def test_resume_continues_the_same_session_from_stored_messages(
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
    paused_request = agent_requests(fake_anthropic)[0]
    # A fresh model client and token store: what a restarted app would have.
    model = FakeAnthropic(
        script=[
            tool_turn(tool_call("get_ticket", {"ticket_id": "T-1002"}, "tu_check")),
            text_reply("T-1002 is closed now."),
        ]
    )
    deps = make_deps(seeded_database_url, model, fake_discord, approvals, TokenStore())

    await handle_click(deps, paused.request(), "approve", AVERY)

    sid = paused.session_id
    assert db.execute("SELECT count(*) FROM sessions").fetchone() == (1,)
    assert db.execute("SELECT id FROM sessions").fetchone() == (sid,)
    assert seqs(db, sid) == [0, 1, 2, 3, 4, 5]
    stored = [
        r[0]
        for r in db.execute(
            "SELECT content FROM messages WHERE session_id = %s ORDER BY seq", (sid,)
        )
    ]
    first = model.requests[0]
    # The resumed request is the stored conversation plus the one new results message.
    assert [m["content"] for m in first.messages] == stored[:3]
    assert first.messages[0] == paused_request.messages[0]
    assert first.system == paused_request.system
    assert [r["tool_use_id"] for r in tool_results(stored[2])] == ["tu_close"]
    # The iteration count carries on: 1 before the pause, then 2 and 3.
    assert session_status(db, sid)[:2] == ("completed", 3)
    iterations = [
        r[0]
        for r in db.execute(
            "SELECT DISTINCT iteration FROM traces WHERE session_id = %s ORDER BY 1", (sid,)
        )
    ]
    assert iterations == [0, 1, 2, 3]


@pytest.mark.anyio
async def test_running_the_resume_again_changes_nothing(
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
    fake_anthropic.script = [text_reply("Closed.")]
    await handle_click(paused.deps, paused.request(), "approve", AVERY)
    before = (seqs(db, paused.session_id), ticket(db, "T-1002"), len(fake_discord.posts))

    async with await psycopg.AsyncConnection.connect(seeded_database_url, autocommit=True) as c:
        await resume_session(c, paused.deps, paused.session_id)

    after = (seqs(db, paused.session_id), ticket(db, "T-1002"), len(fake_discord.posts))
    assert after == before
    assert asker_replies(fake_discord) == ["Closed."]


@pytest.mark.anyio
async def test_reply_goes_out_by_rest_after_15_minutes(
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
    db.execute(
        "UPDATE sessions SET created_at = now() - interval '16 minutes' WHERE id = %s",
        (paused.session_id,),
    )
    edits_before = list(fake_discord.edits)
    fake_anthropic.script = [text_reply("T-1002 is closed.")]

    await handle_click(paused.deps, paused.request(), "approve", AVERY)

    assert fake_discord.edits == edits_before  # only the waiting note, never edited again
    (reply,) = fake_discord.posts_in(ASK_CHANNEL)
    assert reply.content == "T-1002 is closed."
    assert reply.allowed_mentions == {"parse": []}
    assert reply.components is None
    assert session_status(db, paused.session_id) == ("completed", 2, None)


@pytest.mark.anyio
async def test_reply_edits_the_original_response_within_15_minutes(
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
    db.execute(
        "UPDATE sessions SET created_at = now() - interval '10 minutes' WHERE id = %s",
        (paused.session_id,),
    )
    fake_anthropic.script = [text_reply("T-1002 is closed.")]

    await handle_click(paused.deps, paused.request(), "approve", AVERY)

    assert fake_discord.edits[-1] == ("app", "tok", "T-1002 is closed.")
    assert fake_discord.posts_in(ASK_CHANNEL) == []


@pytest.mark.anyio
async def test_failed_rest_post_records_reply_error_and_notifies_the_approvers(
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
    request = httpx.Request("POST", f"https://discord.com/api/v10/channels/{ASK_CHANNEL}/messages")
    fake_discord.fail_post_in[ASK_CHANNEL] = httpx.HTTPStatusError(
        "403 Forbidden", request=request, response=httpx.Response(403, request=request)
    )
    deps = make_deps(seeded_database_url, fake_anthropic, fake_discord, approvals, TokenStore())
    fake_anthropic.script = [text_reply("T-1002 is closed.")]

    await handle_click(deps, paused.request(), "approve", AVERY)

    status, _, reply_error = session_status(db, paused.session_id)
    assert status == "completed"
    assert reply_error and "403" in reply_error
    notices = [p for p in fake_discord.posts_in(APPROVER_CHANNEL) if not p.components]
    (notice,) = notices
    assert f"<@{ASKER}>" in notice.content
    assert str(paused.session_id) in notice.content
    assert notice.allowed_mentions == {"parse": [], "users": [ASKER]}


@pytest.mark.anyio
async def test_a_resumed_turn_can_pause_again(
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
        tool_call("close_ticket", {"ticket_id": "T-1002"}, "tu_one"),
    )
    fake_anthropic.script = [
        tool_turn(tool_call("close_ticket", {"ticket_id": "T-1005"}, "tu_two")),
        text_reply("Both handled."),
    ]

    await handle_click(paused.deps, paused.request("tu_one"), "approve", AVERY)

    (second_id,) = db.execute(
        "SELECT id FROM approval_requests WHERE tool_use_id = 'tu_two'"
    ).fetchone() or (None,)
    assert db.execute(
        "SELECT turn, status FROM approval_requests WHERE id = %s", (second_id,)
    ).fetchone() == (2, "pending")
    assert session_status(db, paused.session_id)[0] == "awaiting_approval"
    assert asker_replies(fake_discord) == []

    await handle_click(paused.deps, second_id, "approve", AVERY)

    assert asker_replies(fake_discord) == ["Both handled."]
    assert ticket(db, "T-1005")[0] == "closed"
    assert seqs(db, paused.session_id) == [0, 1, 2, 3, 4, 5]
    assert session_status(db, paused.session_id)[0] == "completed"
    stored_results = json.dumps(
        db.execute(
            "SELECT content FROM messages WHERE session_id = %s AND seq = 4", (paused.session_id,)
        ).fetchone()
    )
    assert "tu_two" in stored_results
