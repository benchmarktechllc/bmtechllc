"""Shared fixtures: a fresh local Postgres per test, a mocked Discord REST client,
a scripted fake Anthropic client, and an Ed25519 signer standing in for Discord."""

from __future__ import annotations

import copy
import json
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import anthropic
import httpx2
import psycopg
import pytest
from anthropic.types import Message
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
from psycopg import sql
from psycopg.conninfo import make_conninfo

from triage_agent.agent.client import ModelRequest
from triage_agent.app import create_app
from triage_agent.config import Settings
from triage_agent.db.migrate import apply_migrations
from triage_agent.db.seed import apply_seed

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


@pytest.fixture
def seeded_database_url(database_url: str) -> str:
    """`database_url` with seed/synthetic.sql loaded (tickets, KB articles, teammates)."""
    with psycopg.connect(database_url) as conn:
        apply_seed(conn)
    return database_url


def insert_test_session(
    db: psycopg.Connection[tuple[Any, ...]], question: str = "what's the status of T-1002?"
) -> UUID:
    row = db.execute(
        "INSERT INTO sessions (interaction_id, guild_id, channel_id, asker_user_id, question)"
        " VALUES (%s, 'g', 'c', 'u', %s) RETURNING id",
        (f"i-{uuid4().hex}", question),
    ).fetchone()
    assert row is not None
    session_id: UUID = row[0]
    return session_id


# --- Anthropic ----------------------------------------------------------------


@dataclass
class FakeAnthropic:
    """Scripted stand-in for the model client: returns (or raises) the scripted items in order
    and records a deep copy of every request. Never touches the network."""

    script: list[Message | Exception] = field(default_factory=list)
    requests: list[ModelRequest] = field(default_factory=list)
    on_call: Callable[[ModelRequest], None] | None = None

    async def create(self, request: ModelRequest) -> Message:
        self.requests.append(copy.deepcopy(request))
        if self.on_call is not None:
            self.on_call(request)
        if not self.script:
            raise AssertionError(f"FakeAnthropic: no scripted response left for {request.model}")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def fake_anthropic() -> FakeAnthropic:
    return FakeAnthropic()


def make_message(
    content: list[dict[str, Any]],
    stop_reason: str = "end_turn",
    model: str = "claude-sonnet-5-5",
    input_tokens: int = 100,
    output_tokens: int = 20,
) -> Message:
    return Message.model_validate(
        {
            "id": f"msg_{uuid4().hex[:12]}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }
    )


def text_reply(text: str, stop_reason: str = "end_turn") -> Message:
    return make_message([{"type": "text", "text": text}], stop_reason=stop_reason)


def tool_call(
    name: str, tool_input: dict[str, Any], tool_use_id: str | None = None
) -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": tool_use_id or f"toolu_{uuid4().hex[:12]}",
        "name": name,
        "input": tool_input,
    }


def tool_turn(*calls: dict[str, Any]) -> Message:
    return make_message(list(calls), stop_reason="tool_use")


def classifier_reply(
    concerns: list[tuple[str, str]],
    primary: int = 0,
    confidence: float = 0.9,
    more_concerns: bool = False,
) -> Message:
    body = {
        "concerns": [{"summary": s, "category": c} for s, c in concerns],
        "primary": primary,
        "confidence": confidence,
        "more_concerns": more_concerns,
    }
    return make_message(
        [{"type": "text", "text": json.dumps(body)}],
        model="claude-haiku-5-5",
        input_tokens=300,
        output_tokens=60,
    )


_FAKE_REQUEST = httpx2.Request("POST", "https://api.anthropic.invalid/v1/messages")


def api_error() -> anthropic.APIError:
    return anthropic.InternalServerError(
        "overloaded", response=httpx2.Response(500, request=_FAKE_REQUEST), body=None
    )


def api_timeout() -> anthropic.APIError:
    return anthropic.APITimeoutError(request=_FAKE_REQUEST)


# --- Discord ----------------------------------------------------------------


@dataclass
class FakeDiscordRest:
    """Mocked Discord REST client. Records every call; never touches the network."""

    edits: list[tuple[str, str, str]] = field(default_factory=list)
    fail_with: Exception | None = None

    async def edit_original_response(
        self, application_id: str, interaction_token: str, content: str
    ) -> None:
        self.edits.append((application_id, interaction_token, content))
        if self.fail_with is not None:
            raise self.fail_with


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
def client(
    settings: Settings, fake_discord: FakeDiscordRest, fake_anthropic: FakeAnthropic
) -> Iterator[TestClient]:
    app = create_app(settings=settings, discord=fake_discord, model=fake_anthropic)
    with TestClient(app) as c:
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
