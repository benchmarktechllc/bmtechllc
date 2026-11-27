"""Approve/Deny clicks (SPEC.md section 5, Click): the role check, the type 6 ACK, each status
transition, self-approval, the decided message, and the double-click race."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import anyio
import psycopg
import pytest
from fastapi.testclient import TestClient

from tests.conftest import (
    APPROVER_CHANNEL,
    APPROVER_ROLE,
    ASKER,
    AVERY,
    BLAKE,
    DiscordSigner,
    FakeAnthropic,
    FakeDiscordRest,
    ask_payload,
    classifier_reply,
    component_payload,
    encode,
    text_reply,
    tool_call,
    tool_turn,
)
from tests.flows import (
    approval_posts,
    asker_replies,
    last_user_results,
    pause,
    receipts,
    request_row,
    session_status,
    ticket,
)
from triage_agent.agent.resume import handle_click
from triage_agent.app import create_app
from triage_agent.config import ApprovalConfig, Settings
from triage_agent.db.approvals import decide_request

Db = psycopg.Connection[tuple[Any, ...]]

NOT_AUTHORIZED = {
    "type": 4,
    "data": {"content": "You're not authorized to approve or deny actions.", "flags": 64},
}


@pytest.fixture(autouse=True)
def _seeded(seeded_database_url: str) -> None:
    """The app-level tests share `database_url` with the `client` fixture; seed its tickets."""


def post(client: TestClient, signer: DiscordSigner, payload: dict[str, Any]) -> Any:
    body = encode(payload)
    return client.post("/interactions", content=body, headers=signer.headers(body))


@dataclass
class AskedSession:
    session_id: UUID
    requests: dict[str, UUID]


def ask_until_paused(
    client: TestClient,
    signer: DiscordSigner,
    fake_anthropic: FakeAnthropic,
    db: Db,
    *calls: dict[str, Any],
    after: list[Any] | None = None,
) -> AskedSession:
    """/ask through the app until it pauses on `calls`. `after` is scripted for the resume."""
    fake_anthropic.script = [
        classifier_reply([("close ticket T-1002", "other")]),
        tool_turn(*(calls or (tool_call("close_ticket", {"ticket_id": "T-1002"}, "tu_close"),))),
        *(after if after is not None else [text_reply("T-1002 is now closed.")]),
    ]
    assert post(client, signer, ask_payload(question="please close T-1002")).json() == {"type": 5}
    (sid,) = db.execute("SELECT id FROM sessions").fetchone() or (None,)
    rows = db.execute(
        "SELECT tool_use_id, id FROM approval_requests WHERE session_id = %s", (sid,)
    ).fetchall()
    assert session_status(db, sid)[0] == "awaiting_approval"
    return AskedSession(sid, {r[0]: r[1] for r in rows})


# --- the role check ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("user_id", "roles"),
    [
        (BLAKE, []),  # a non-approver
        (ASKER, []),  # the asker, without the role
        (AVERY, ["1800000000000000999"]),  # some other role
    ],
    ids=["non_approver", "asker_without_role", "other_role"],
)
def test_click_without_the_approver_role_is_rejected_and_changes_nothing(
    user_id: str,
    roles: list[str],
    client: TestClient,
    signer: DiscordSigner,
    fake_anthropic: FakeAnthropic,
    fake_discord: FakeDiscordRest,
    db: Db,
) -> None:
    asked = ask_until_paused(client, signer, fake_anthropic, db, after=[])
    rid = asked.requests["tu_close"]
    edits_before = list(fake_discord.edits)
    requests_before = len(fake_anthropic.requests)

    resp = post(client, signer, component_payload(rid, "approve", user_id=user_id, roles=roles))

    assert resp.status_code == 200
    assert resp.json() == NOT_AUTHORIZED
    req = request_row(db, rid)
    assert (req["status"], req["decided_by"], req["decided_at"]) == ("pending", None, None)
    assert session_status(db, asked.session_id)[0] == "awaiting_approval"
    assert fake_discord.channel_edits == []
    assert fake_discord.edits == edits_before
    assert len(fake_anthropic.requests) == requests_before
    assert receipts(db, asked.session_id) == []


def test_click_from_a_dm_without_member_roles_is_rejected(
    client: TestClient, signer: DiscordSigner, fake_anthropic: FakeAnthropic, db: Db
) -> None:
    asked = ask_until_paused(client, signer, fake_anthropic, db, after=[])
    payload = component_payload(asked.requests["tu_close"])
    payload["user"] = payload.pop("member")["user"]
    assert post(client, signer, payload).json() == NOT_AUTHORIZED
    assert request_row(db, asked.requests["tu_close"])["status"] == "pending"


@pytest.mark.parametrize(
    "custom_id", ["approval:not-a-uuid:approve", "approval::approve", "other:1:approve", ""]
)
def test_malformed_custom_id_is_rejected(
    custom_id: str, client: TestClient, signer: DiscordSigner
) -> None:
    payload = component_payload(UUID(int=1))
    payload["data"]["custom_id"] = custom_id
    assert post(client, signer, payload).status_code == 400


def test_unknown_action_is_rejected(client: TestClient, signer: DiscordSigner) -> None:
    assert post(client, signer, component_payload(UUID(int=1), "maybe")).status_code == 400


def test_click_on_an_unknown_request_is_acked_and_changes_nothing(
    client: TestClient, signer: DiscordSigner, fake_discord: FakeDiscordRest
) -> None:
    assert post(client, signer, component_payload(UUID(int=1))).json() == {"type": 6}
    assert fake_discord.channel_edits == []


# --- decisions through the app ------------------------------------------------------


def test_approve_click_closes_the_ticket_and_replies_once(
    client: TestClient,
    signer: DiscordSigner,
    fake_anthropic: FakeAnthropic,
    fake_discord: FakeDiscordRest,
    db: Db,
) -> None:
    asked = ask_until_paused(client, signer, fake_anthropic, db)
    rid = asked.requests["tu_close"]
    (button_post,) = approval_posts(fake_discord)

    resp = post(client, signer, component_payload(rid, "approve", user_id=AVERY))

    assert resp.json() == {"type": 6}
    req = request_row(db, rid)
    assert req["status"] == "approved"
    assert req["decided_by"] == AVERY
    assert req["self_approved"] is False
    assert req["decided_at"] is not None
    # The approval message now shows the outcome, who and when, with no buttons.
    ((channel, message_id, content, components),) = fake_discord.channel_edits
    assert (channel, message_id, components) == (APPROVER_CHANNEL, button_post.message_id, [])
    assert "Approved" in content and f"<@{AVERY}>" in content and "UTC" in content
    # Resumed: the ticket closed once, one receipt, and one reply (an edit, within 15 min).
    assert ticket(db, "T-1002")[0] == "closed"
    assert len(receipts(db, asked.session_id)) == 1
    assert asker_replies(fake_discord) == ["T-1002 is now closed."]
    assert fake_discord.edits[-1] == (
        "1200000000000000001",
        "interaction-token-abc",
        "T-1002 is now closed.",
    )
    assert session_status(db, asked.session_id)[0] == "completed"
    assert fake_anthropic.script == []


def test_deny_click_tells_the_model_who_denied_and_closes_nothing(
    client: TestClient,
    signer: DiscordSigner,
    fake_anthropic: FakeAnthropic,
    fake_discord: FakeDiscordRest,
    db: Db,
) -> None:
    asked = ask_until_paused(
        client, signer, fake_anthropic, db, after=[text_reply("The close was denied.")]
    )
    rid = asked.requests["tu_close"]

    assert post(client, signer, component_payload(rid, "deny", user_id=BLAKE)).json() == {"type": 6}

    req = request_row(db, rid)
    assert (req["status"], req["decided_by"], req["self_approved"]) == ("denied", BLAKE, False)
    (result,) = last_user_results(fake_anthropic)
    assert result == {
        "type": "tool_result",
        "tool_use_id": "tu_close",
        "content": "denied by Blake Synthetic",
        "is_error": True,
    }
    assert ticket(db, "T-1002")[0] == "open"
    assert receipts(db, asked.session_id) == []
    ((_, _, content, components),) = fake_discord.channel_edits
    assert "Denied" in content and f"<@{BLAKE}>" in content and components == []
    assert asker_replies(fake_discord) == ["The close was denied."]


def test_asker_with_the_role_can_approve_their_own_request(
    client: TestClient,
    signer: DiscordSigner,
    fake_anthropic: FakeAnthropic,
    db: Db,
) -> None:
    asked = ask_until_paused(client, signer, fake_anthropic, db)
    rid = asked.requests["tu_close"]

    resp = post(
        client, signer, component_payload(rid, "approve", user_id=ASKER, roles=[APPROVER_ROLE])
    )

    assert resp.json() == {"type": 6}
    req = request_row(db, rid)
    assert (req["status"], req["decided_by"], req["self_approved"]) == ("approved", ASKER, True)
    assert ticket(db, "T-1002")[0] == "closed"


def test_click_on_an_already_decided_request_changes_nothing(
    client: TestClient,
    signer: DiscordSigner,
    fake_anthropic: FakeAnthropic,
    fake_discord: FakeDiscordRest,
    db: Db,
) -> None:
    asked = ask_until_paused(client, signer, fake_anthropic, db)
    rid = asked.requests["tu_close"]
    post(client, signer, component_payload(rid, "approve", user_id=AVERY))
    decided = request_row(db, rid)
    edits, channel_edits = list(fake_discord.edits), list(fake_discord.channel_edits)
    model_calls = len(fake_anthropic.requests)

    for action in ("deny", "approve"):
        resp = post(client, signer, component_payload(rid, action, user_id=BLAKE))
        assert resp.json() == {"type": 6}

    assert request_row(db, rid) == decided
    assert fake_discord.edits == edits
    assert fake_discord.channel_edits == channel_edits
    assert len(fake_anthropic.requests) == model_calls
    assert len(receipts(db, asked.session_id)) == 1


# --- each status transition, at the database level -----------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("first", ["approved", "denied", "expired"])
async def test_status_transitions_happen_once_from_pending(
    first: str,
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
    rid = paused.request("tu_a")
    decider = None if first == "expired" else AVERY
    async with await psycopg.AsyncConnection.connect(seeded_database_url, autocommit=True) as c:
        decision = await decide_request(c, rid, first, decider)  # type: ignore[arg-type]
        assert decision.changed is True
        assert decision.resume is False  # tu_b is still pending
        assert decision.request is not None and decision.request.status == first
        assert session_status(db, paused.session_id)[0] == "awaiting_approval"

        for later in ("approved", "denied", "expired"):
            again = await decide_request(c, rid, later, BLAKE)  # type: ignore[arg-type]
            assert (again.changed, again.resume) == (False, False)

    req = request_row(db, rid)
    assert req["status"] == first
    assert req["decided_by"] == decider
    assert req["decided_at"] is not None


@pytest.mark.anyio
async def test_last_decision_of_the_turn_flips_the_session_to_running(
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
    db.execute(
        "UPDATE sessions SET heartbeat_at = now() - interval '1 hour' WHERE id = %s",
        (paused.session_id,),
    )
    async with await psycopg.AsyncConnection.connect(seeded_database_url, autocommit=True) as c:
        first = await decide_request(c, paused.request("tu_a"), "approved", AVERY)
        second = await decide_request(c, paused.request("tu_b"), "denied", AVERY)
    assert (first.resume, second.resume) == (False, True)
    (status, fresh) = db.execute(
        "SELECT status, heartbeat_at > now() - interval '1 minute' FROM sessions WHERE id = %s",
        (paused.session_id,),
    ).fetchone() or (None, None)
    assert (status, fresh) == ("running", True)


# --- the ACK and the race ------------------------------------------------------------


@pytest.mark.anyio
async def test_role_holders_click_is_acked_with_type_6_before_the_decision_runs(
    settings: Settings,
    signer: DiscordSigner,
    seeded_database_url: str,
    db: Db,
    fake_anthropic: FakeAnthropic,
    approvals: ApprovalConfig,
) -> None:
    events: list[str] = []

    class SlowDiscord(FakeDiscordRest):
        async def edit_channel_message(
            self,
            channel_id: str,
            message_id: str,
            content: str,
            *,
            components: list[dict[str, Any]],
        ) -> None:
            events.append("edit-start")
            await anyio.sleep(0.5)
            events.append("edit-done")
            await super().edit_channel_message(
                channel_id, message_id, content, components=components
            )

    discord = SlowDiscord()
    paused = await pause(
        seeded_database_url,
        db,
        fake_anthropic,
        discord,
        approvals,
        tool_call("close_ticket", {"ticket_id": "T-1002"}, "tu_close"),
    )
    fake_anthropic.script = [text_reply("Closed.")]
    app = create_app(settings=settings, discord=discord, model=fake_anthropic)

    body = encode(component_payload(paused.request(), "approve", user_id=AVERY))
    headers = [(k.lower().encode(), v.encode()) for k, v in signer.headers(body).items()]
    headers.append((b"content-length", str(len(body)).encode()))
    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/interactions",
        "raw_path": b"/interactions",
        "root_path": "",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    }
    sent_body = False

    async def receive() -> dict[str, Any]:
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": body, "more_body": False}
        await anyio.sleep_forever()
        raise AssertionError("unreachable")

    response_body = b""
    ack_at = 0.0
    start = time.monotonic()

    async def send(message: dict[str, Any]) -> None:
        nonlocal response_body, ack_at
        if message["type"] == "http.response.body":
            response_body += message.get("body", b"")
            if not message.get("more_body", False):
                ack_at = time.monotonic() - start
                events.append("ack-sent")

    with anyio.fail_after(10):
        await app(scope, receive, send)  # type: ignore[arg-type]

    assert json.loads(response_body) == {"type": 6}
    assert ack_at < 3.0
    assert events == ["ack-sent", "edit-start", "edit-done"]
    assert request_row(db, paused.request())["status"] == "approved"


@pytest.mark.anyio
async def test_two_concurrent_approve_clicks_close_once(
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
    fake_anthropic.script = [text_reply("Closed T-1002.")]

    await asyncio.gather(
        handle_click(paused.deps, paused.request(), "approve", AVERY),
        handle_click(paused.deps, paused.request(), "approve", BLAKE),
    )

    assert len(receipts(db, paused.session_id)) == 1
    assert ticket(db, "T-1002") == ("closed", notes_before + 1)
    assert len(fake_discord.channel_edits) == 1
    assert asker_replies(fake_discord) == ["Closed T-1002."]
    assert request_row(db, paused.request())["decided_by"] in (AVERY, BLAKE)
    assert fake_anthropic.script == []
    assert session_status(db, paused.session_id)[0] == "completed"
