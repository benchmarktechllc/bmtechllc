"""Helpers shared by the Slice 3 tests: drive a session to a paused approval turn through the
real handle_ask, then read back what the database and the fakes recorded."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import psycopg
from anthropic.types import Message

from tests.conftest import (
    APPROVER_CHANNEL,
    ASKER,
    FakeAnthropic,
    FakeDiscordRest,
    classifier_reply,
    insert_test_session,
    tool_turn,
)
from triage_agent.agent.deps import Deps
from triage_agent.agent.pipeline import AskJob, handle_ask
from triage_agent.agent.reply import TokenStore
from triage_agent.config import ApprovalConfig

Db = psycopg.Connection[tuple[Any, ...]]

ASK_CHANNEL = "1400000000000000001"
CLASSIFY_CLOSE = classifier_reply([("close ticket T-1002", "other")])


@dataclass
class Paused:
    session_id: UUID
    requests: dict[str, UUID]  # tool_use_id -> approval request id
    deps: Deps

    def request(self, tool_use_id: str = "tu_close") -> UUID:
        return self.requests[tool_use_id]


def make_deps(
    database_url: str,
    model: FakeAnthropic,
    discord: FakeDiscordRest,
    approvals: ApprovalConfig,
    tokens: TokenStore | None = None,
) -> Deps:
    return Deps(
        database_url=database_url,
        model=model,
        discord=discord,
        tokens=tokens if tokens is not None else TokenStore(),
        approvals=approvals,
    )


async def pause(
    database_url: str,
    db: Db,
    model: FakeAnthropic,
    discord: FakeDiscordRest,
    approvals: ApprovalConfig,
    *calls: dict[str, Any],
    question: str = "please close T-1002",
    before: list[Message] | None = None,
    turn: Message | None = None,
    asker: str = ASKER,
) -> Paused:
    """Run /ask until the loop pauses on `calls` (one tool_use turn), after the optional
    `before` turns. Leaves the session awaiting approval, with the fake script empty."""
    session_id = insert_test_session(db, question, asker=asker, channel=ASK_CHANNEL)
    tokens = TokenStore()
    model.script = [CLASSIFY_CLOSE, *(before or []), turn or tool_turn(*calls)]
    await handle_ask(
        database_url,
        model,
        discord,
        AskJob(session_id=session_id, question=question, application_id="app", token="tok"),
        tokens=tokens,
        approvals=approvals,
    )
    assert model.script == [], "the pause should consume every scripted response"
    rows = db.execute(
        "SELECT tool_use_id, id FROM approval_requests WHERE session_id = %s", (session_id,)
    ).fetchall()
    return Paused(
        session_id=session_id,
        requests={r[0]: r[1] for r in rows},
        deps=make_deps(database_url, model, discord, approvals, tokens),
    )


def request_row(db: Db, request_id: UUID) -> dict[str, Any]:
    cur = db.execute(
        "SELECT status, turn, tool_use_id, tool_name, tool_input, requested_by, decided_by,"
        " self_approved, decided_at, expires_at, discord_message_id"
        " FROM approval_requests WHERE id = %s",
        (request_id,),
    )
    row = cur.fetchone()
    assert row is not None
    names = [d.name for d in cur.description or []]
    return dict(zip(names, row, strict=True))


def session_status(db: Db, session_id: UUID) -> tuple[Any, ...]:
    row = db.execute(
        "SELECT status, iterations, reply_error FROM sessions WHERE id = %s", (session_id,)
    ).fetchone()
    assert row is not None
    return row


def receipts(db: Db, session_id: UUID) -> list[tuple[Any, ...]]:
    return db.execute(
        "SELECT approval_request_id, tool_name, tool_input, outcome, result FROM receipts"
        " WHERE session_id = %s ORDER BY executed_at",
        (session_id,),
    ).fetchall()


def ticket(db: Db, ticket_id: str) -> tuple[str, int]:
    """(status, number of history notes)."""
    row = db.execute(
        "SELECT status, jsonb_array_length(history) FROM tickets WHERE id = %s", (ticket_id,)
    ).fetchone()
    assert row is not None
    return row[0], row[1]


def seqs(db: Db, session_id: UUID) -> list[int]:
    return [
        r[0]
        for r in db.execute(
            "SELECT seq FROM messages WHERE session_id = %s ORDER BY seq", (session_id,)
        )
    ]


def tool_results(content: object) -> list[dict[str, Any]]:
    assert isinstance(content, list)
    return [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]


def last_user_results(model: FakeAnthropic) -> list[dict[str, Any]]:
    """The tool_result blocks of the last agent request's final (user) message."""
    agent = [r for r in model.requests if r.model == "claude-sonnet-5-5"]
    return tool_results(agent[-1].messages[-1]["content"])


def asker_replies(discord: FakeDiscordRest) -> list[str]:
    """Every answer the asker saw: original-response edits that are not the waiting note,
    plus bot-token posts to the asker's channel."""
    edits = [e[2] for e in discord.edits if not e[2].startswith("Waiting for approval")]
    return edits + [p.content for p in discord.posts_in(ASK_CHANNEL)]


def approval_posts(discord: FakeDiscordRest) -> list[Any]:
    return [p for p in discord.posts_in(APPROVER_CHANNEL) if p.components]


def custom_ids(components: list[dict[str, Any]] | None) -> list[str]:
    assert components
    return [c["custom_id"] for row in components for c in row["components"]]


def result_json(block: dict[str, Any]) -> Any:
    content = block["content"]
    return json.loads(content) if isinstance(content, str) else content
