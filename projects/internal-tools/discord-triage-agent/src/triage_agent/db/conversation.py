"""messages (append-only conversation) and traces (per-call audit) access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

Role = Literal["user", "assistant"]
HookDecision = Literal["allow", "require_approval", "deny"]
JsonObject = dict[str, object]


@dataclass(frozen=True)
class StoredMessage:
    seq: int
    role: Role
    content: list[JsonObject]


async def append_message(
    conn: psycopg.AsyncConnection[Any],
    session_id: UUID,
    role: Role,
    content: list[JsonObject],
) -> int:
    """Append one message at the next seq and return that seq. Never updates a row.

    One writer per session; the UNIQUE (session_id, seq) constraint rejects a racing duplicate.
    """
    cur = await conn.execute(
        """
        INSERT INTO messages (session_id, seq, role, content)
        SELECT %s, coalesce(max(seq) + 1, 0), %s, %s FROM messages WHERE session_id = %s
        RETURNING seq
        """,
        (session_id, role, Jsonb(content), session_id),
    )
    row = await cur.fetchone()
    if row is None:  # pragma: no cover - INSERT ... SELECT with an aggregate always yields a row
        raise RuntimeError("append_message inserted nothing")
    seq: int = row[0]
    return seq


async def load_messages(
    conn: psycopg.AsyncConnection[Any], session_id: UUID
) -> list[StoredMessage]:
    cur = await conn.execute(
        "SELECT seq, role, content FROM messages WHERE session_id = %s ORDER BY seq",
        (session_id,),
    )
    out: list[StoredMessage] = []
    for seq, role, content in await cur.fetchall():
        if role not in ("user", "assistant") or not isinstance(content, list):
            raise ValueError(f"malformed message seq={seq} in session {session_id}")
        # jsonb arrays of objects, written only by append_message.
        out.append(StoredMessage(seq=seq, role=role, content=cast(list[JsonObject], content)))
    return out


@dataclass(frozen=True)
class Trace:
    session_id: UUID
    iteration: int
    model: str
    stop_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    tool_use_id: str | None = None
    tool_name: str | None = None
    tool_input: object | None = None
    tool_result: object | None = None
    hook_decision: HookDecision | None = None
    latency_ms: int | None = None
    error: str | None = None


async def insert_trace(conn: psycopg.AsyncConnection[Any], t: Trace) -> None:
    await conn.execute(
        """
        INSERT INTO traces (session_id, iteration, model, stop_reason, input_tokens,
                            output_tokens, tool_use_id, tool_name, tool_input, tool_result,
                            hook_decision, latency_ms, error)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            t.session_id,
            t.iteration,
            t.model,
            t.stop_reason,
            t.input_tokens,
            t.output_tokens,
            t.tool_use_id,
            t.tool_name,
            None if t.tool_input is None else Jsonb(t.tool_input),
            None if t.tool_result is None else Jsonb(t.tool_result),
            t.hook_decision,
            t.latency_ms,
            t.error,
        ),
    )
