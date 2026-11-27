"""POST /interactions must reject anything not signed by Discord's key (401)."""

from __future__ import annotations

from typing import Any

import psycopg
from fastapi.testclient import TestClient
from nacl.signing import SigningKey

from tests.conftest import DiscordSigner, FakeDiscordRest, ask_payload, encode
from triage_agent.discord.verify import verify_signature

PING = encode({"type": 1})


def test_valid_signature_is_accepted(client: TestClient, signer: DiscordSigner) -> None:
    resp = client.post("/interactions", content=PING, headers=signer.headers(PING))
    assert resp.status_code == 200


def test_signature_from_another_key_is_rejected(client: TestClient) -> None:
    other = DiscordSigner(SigningKey.generate())
    resp = client.post("/interactions", content=PING, headers=other.headers(PING))
    assert resp.status_code == 401


def test_missing_signature_headers_are_rejected(client: TestClient) -> None:
    resp = client.post("/interactions", content=PING, headers={"Content-Type": "application/json"})
    assert resp.status_code == 401


def test_missing_timestamp_is_rejected(client: TestClient, signer: DiscordSigner) -> None:
    headers = signer.headers(PING)
    del headers["X-Signature-Timestamp"]
    resp = client.post("/interactions", content=PING, headers=headers)
    assert resp.status_code == 401


def test_malformed_signature_hex_is_rejected(client: TestClient, signer: DiscordSigner) -> None:
    headers = signer.headers(PING)
    headers["X-Signature-Ed25519"] = "not-hex"
    resp = client.post("/interactions", content=PING, headers=headers)
    assert resp.status_code == 401


def test_body_tampered_after_signing_is_rejected(client: TestClient, signer: DiscordSigner) -> None:
    headers = signer.headers(PING)
    resp = client.post("/interactions", content=encode({"type": 2}), headers=headers)
    assert resp.status_code == 401


def test_rejected_request_records_nothing_and_calls_no_discord(
    client: TestClient,
    fake_discord: FakeDiscordRest,
    db: psycopg.Connection[tuple[Any, ...]],
) -> None:
    body = encode(ask_payload())
    other = DiscordSigner(SigningKey.generate())
    resp = client.post("/interactions", content=body, headers=other.headers(body))
    assert resp.status_code == 401
    assert fake_discord.edits == []
    row = db.execute("SELECT count(*) FROM sessions").fetchone()
    assert row == (0,)


def test_verify_signature_unit(signer: DiscordSigner) -> None:
    body = b'{"type":1}'
    headers = signer.headers(body, timestamp="1700000000")
    sig = headers["X-Signature-Ed25519"]
    assert verify_signature(signer.public_key_hex, sig, "1700000000", body)
    assert not verify_signature(signer.public_key_hex, sig, "1700000001", body)
    assert not verify_signature(signer.public_key_hex, None, "1700000000", body)
    assert not verify_signature(signer.public_key_hex, sig, None, body)
    assert not verify_signature(signer.public_key_hex, "zz", "1700000000", body)
    assert not verify_signature("not-a-key", sig, "1700000000", body)
