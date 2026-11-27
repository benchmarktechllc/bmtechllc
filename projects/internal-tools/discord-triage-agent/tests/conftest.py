"""Shared fixtures: a fresh local Postgres per test, a mocked Discord REST client,
and an Ed25519 signer standing in for Discord."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import psycopg
import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
from psycopg import sql
from psycopg.conninfo import make_conninfo

from triage_agent.app import create_app
from triage_agent.config import Settings
from triage_agent.db.migrate import apply_migrations

load_dotenv()  # picks up TEST_DATABASE_URL from .env if it is set there

TEST_DB_HELP = (
    "TEST_DATABASE_URL is not set. Start the local test Postgres with:\n"
    "  docker run -d --name triage-agent-test-pg -e POSTGRES_PASSWORD=postgres "
    "-p 127.0.0.1:54329:5432 postgres:16\n"
    "then: export TEST_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:54329/postgres"
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# --- Postgres -------------------------------------------------------------


@pytest.fixture(scope="session")
def admin_database_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.fail(TEST_DB_HELP, pytrace=False)
    return url


@pytest.fixture
def database_url(admin_database_url: str) -> Iterator[str]:
    """A fresh, empty database with every migration in migrations/ applied."""
    name = f"triage_test_{uuid4().hex[:12]}"
    with psycopg.connect(admin_database_url, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    url = make_conninfo(admin_database_url, dbname=name)
    try:
        with psycopg.connect(url) as conn:
            apply_migrations(conn)
        yield url
    finally:
        with psycopg.connect(admin_database_url, autocommit=True) as admin:
            admin.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
            )


@pytest.fixture
def db(database_url: str) -> Iterator[psycopg.Connection[tuple[Any, ...]]]:
    with psycopg.connect(database_url, autocommit=True) as conn:
        yield conn


# --- Discord ----------------------------------------------------------------


@dataclass
class FakeDiscordRest:
    """Mocked Discord REST client. Records every call; never touches the network."""

    edits: list[tuple[str, str, str]] = field(default_factory=list)

    async def edit_original_response(
        self, application_id: str, interaction_token: str, content: str
    ) -> None:
        self.edits.append((application_id, interaction_token, content))


@pytest.fixture
def fake_discord() -> FakeDiscordRest:
    return FakeDiscordRest()


@dataclass
class DiscordSigner:
    """Signs request bodies the way Discord does (Ed25519 over timestamp + body)."""

    key: SigningKey

    @property
    def public_key_hex(self) -> str:
        return self.key.verify_key.encode(encoder=HexEncoder).decode()

    def headers(self, body: bytes, timestamp: str | None = None) -> dict[str, str]:
        ts = timestamp or str(int(time.time()))
        signature = self.key.sign(ts.encode() + body).signature.hex()
        return {
            "X-Signature-Ed25519": signature,
            "X-Signature-Timestamp": ts,
            "Content-Type": "application/json",
        }


@pytest.fixture
def signer() -> DiscordSigner:
    return DiscordSigner(SigningKey.generate())


@pytest.fixture
def settings(signer: DiscordSigner, database_url: str) -> Settings:
    return Settings(discord_public_key=signer.public_key_hex, database_url=database_url)


@pytest.fixture
def client(settings: Settings, fake_discord: FakeDiscordRest) -> Iterator[TestClient]:
    with TestClient(create_app(settings=settings, discord=fake_discord)) as c:
        yield c


def ask_payload(
    interaction_id: str = "1100000000000000001",
    question: str = "what's the status of T-1002?",
    token: str = "interaction-token-abc",
) -> dict[str, Any]:
    """A minimal APPLICATION_COMMAND interaction for /ask, as Discord sends it in a guild."""
    return {
        "id": interaction_id,
        "application_id": "1200000000000000001",
        "type": 2,
        "token": token,
        "version": 1,
        "guild_id": "1300000000000000001",
        "channel_id": "1400000000000000001",
        "member": {"user": {"id": "1500000000000000001", "username": "asker"}, "roles": []},
        "data": {
            "id": "1600000000000000001",
            "name": "ask",
            "type": 1,
            "options": [{"name": "question", "type": 3, "value": question}],
        },
    }


def encode(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode()
