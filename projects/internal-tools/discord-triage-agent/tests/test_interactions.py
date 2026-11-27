"""PING, /ask routing to a deferred ACK, the reply edit, duplicates and /healthz."""

from __future__ import annotations

import json
import time
from typing import Any
from uuid import UUID

import anyio
import psycopg
import pytest
from fastapi.testclient import TestClient

from tests.conftest import (
    DiscordSigner,
    FakeAnthropic,
    FakeDiscordRest,
    ask_payload,
    classifier_reply,
    encode,
    text_reply,
)
from triage_agent.app import create_app
from triage_agent.config import Settings


def post(client: TestClient, signer: DiscordSigner, payload: dict[str, Any]) -> Any:
    body = encode(payload)
    return client.post("/interactions", content=body, headers=signer.headers(body))


def test_ping_returns_pong(client: TestClient, signer: DiscordSigner) -> None:
    resp = post(client, signer, {"type": 1})
    assert resp.status_code == 200
    assert resp.json() == {"type": 1}


def test_healthz(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def answer(text: str) -> list[Any]:
    """Scripted model responses for one session: a classification, then a final reply."""
    return [classifier_reply([("ticket status", "other")]), text_reply(text)]


def test_ask_gets_deferred_ack_then_the_reply_edit(
    client: TestClient,
    signer: DiscordSigner,
    fake_discord: FakeDiscordRest,
    fake_anthropic: FakeAnthropic,
    db: psycopg.Connection[tuple[Any, ...]],
) -> None:
    fake_anthropic.script = answer("T-1002 is open.")
    start = time.monotonic()
    resp = post(client, signer, ask_payload())
    elapsed = time.monotonic() - start

    assert resp.status_code == 200
    assert resp.json() == {"type": 5}
    assert elapsed < 3.0

    rows = db.execute(
        "SELECT id, interaction_id, guild_id, channel_id, asker_user_id, question, status"
        " FROM sessions"
    ).fetchall()
    assert len(rows) == 1
    session_id, interaction_id, guild_id, channel_id, asker, question, status = rows[0]
    assert isinstance(session_id, UUID)
    assert interaction_id == "1100000000000000001"
    assert guild_id == "1300000000000000001"
    assert channel_id == "1400000000000000001"
    assert asker == "1500000000000000001"
    assert question == "what's the status of T-1002?"
    assert status == "completed"

    assert fake_discord.edits == [
        ("1200000000000000001", "interaction-token-abc", "T-1002 is open.")
    ]


def test_duplicate_interaction_inserts_one_row_and_edits_once(
    client: TestClient,
    signer: DiscordSigner,
    fake_discord: FakeDiscordRest,
    fake_anthropic: FakeAnthropic,
    db: psycopg.Connection[tuple[Any, ...]],
) -> None:
    fake_anthropic.script = answer("once")
    first = post(client, signer, ask_payload())
    second = post(client, signer, ask_payload())

    assert first.json() == {"type": 5}
    assert second.json() == {"type": 5}
    assert db.execute("SELECT count(*) FROM sessions").fetchone() == (1,)
    assert len(fake_discord.edits) == 1


def test_distinct_interactions_get_distinct_sessions(
    client: TestClient,
    signer: DiscordSigner,
    fake_discord: FakeDiscordRest,
    fake_anthropic: FakeAnthropic,
    db: psycopg.Connection[tuple[Any, ...]],
) -> None:
    fake_anthropic.script = [*answer("first"), *answer("second")]
    post(client, signer, ask_payload(interaction_id="1"))
    post(client, signer, ask_payload(interaction_id="2", token="t2"))

    ids = [r[0] for r in db.execute("SELECT id FROM sessions ORDER BY interaction_id")]
    assert len(set(ids)) == 2
    assert [e[2] for e in fake_discord.edits] == ["first", "second"]


def test_dm_ask_uses_top_level_user(
    client: TestClient,
    signer: DiscordSigner,
    db: psycopg.Connection[tuple[Any, ...]],
) -> None:
    payload = ask_payload()
    del payload["member"]
    payload["user"] = {"id": "1700000000000000001", "username": "dm-asker"}
    assert post(client, signer, payload).json() == {"type": 5}
    assert db.execute("SELECT asker_user_id FROM sessions").fetchone() == ("1700000000000000001",)


def test_unknown_command_is_rejected_without_a_session(
    client: TestClient,
    signer: DiscordSigner,
    fake_discord: FakeDiscordRest,
    db: psycopg.Connection[tuple[Any, ...]],
) -> None:
    payload = ask_payload()
    payload["data"]["name"] = "not-ask"
    resp = post(client, signer, payload)
    assert resp.status_code == 400
    assert db.execute("SELECT count(*) FROM sessions").fetchone() == (0,)
    assert fake_discord.edits == []


def test_ask_without_question_is_rejected(
    client: TestClient,
    signer: DiscordSigner,
    db: psycopg.Connection[tuple[Any, ...]],
) -> None:
    payload = ask_payload()
    payload["data"]["options"] = []
    assert post(client, signer, payload).status_code == 400
    assert db.execute("SELECT count(*) FROM sessions").fetchone() == (0,)


def test_signed_body_that_is_not_json_is_rejected(
    client: TestClient, signer: DiscordSigner
) -> None:
    body = b"not json"
    resp = client.post("/interactions", content=body, headers=signer.headers(body))
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_ack_is_sent_before_the_discord_edit_finishes(
    settings: Settings, signer: DiscordSigner
) -> None:
    """The type 5 ACK must not wait on Discord: a slow edit must not delay it."""
    events: list[str] = []

    class SlowDiscord:
        async def edit_original_response(
            self, application_id: str, interaction_token: str, content: str
        ) -> None:
            events.append("edit-start")
            await anyio.sleep(0.5)
            events.append("edit-done")

    model = FakeAnthropic(script=answer("slow"))
    app = create_app(settings=settings, discord=SlowDiscord(), model=model)
    body = encode(ask_payload())
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

    async def send(message: dict[str, Any]) -> None:
        nonlocal response_body
        if message["type"] == "http.response.body":
            response_body += message.get("body", b"")
            if not message.get("more_body", False):
                events.append("ack-sent")

    with anyio.fail_after(5):
        await app(scope, receive, send)  # type: ignore[arg-type]

    assert json.loads(response_body) == {"type": 5}
    assert events == ["ack-sent", "edit-start", "edit-done"]
