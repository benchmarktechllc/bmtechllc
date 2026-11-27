"""approval_requests and receipts access (SPEC.md section 5).

`decide_request` is the one decision transaction, shared by clicks and the expiry pass: lock
the session row, update the request only while it is still pending, and flip the session back
to `running` only when no request of that turn is pending. Only the transaction that does the
flip resumes the session, so double clicks and simultaneous decisions resume exactly once.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

RequestStatus = Literal["pending", "approved", "denied", "expired"]
Decided = Literal["approved", "denied", "expired"]


@dataclass(frozen=True)
class ApprovalRequest:
    id: UUID
    session_id: UUID
    turn: int
    tool_use_id: str
    tool_name: str
    tool_input: dict[str, object]
    status: RequestStatus
    requested_by: str
    decided_by: str | None
    self_approved: bool
    decided_at: datetime | None
    expires_at: datetime
    discord_message_id: str | None


def _request(row: Sequence[Any]) -> ApprovalRequest:
    return ApprovalRequest(*row)


@dataclass(frozen=True)
class Decision:
    changed: bool  # this call decided the request (False: unknown or already decided)
    resume: bool  # this call flipped the session back to running: it must resume it
    request: ApprovalRequest | None


async def create_pending(
    conn: psycopg.AsyncConnection[Any],
    *,
    session_id: UUID,
    turn: int,
    tool_use_id: str,
    tool_name: str,
    tool_input: object,
    requested_by: str,
    ttl: timedelta,
) -> UUID:
    cur = await conn.execute(
        "INSERT INTO approval_requests (session_id, turn, tool_use_id, tool_name, tool_input,"
        " requested_by, expires_at) VALUES (%s, %s, %s, %s, %s, %s, now() + %s) RETURNING id",
        (session_id, turn, tool_use_id, tool_name, Jsonb(tool_input), requested_by, ttl),
    )
    row = await cur.fetchone()
    if row is None:  # pragma: no cover - INSERT ... RETURNING always yields the row
        raise RuntimeError("create_pending inserted nothing")
    request_id: UUID = row[0]
    return request_id


async def set_message_id(
    conn: psycopg.AsyncConnection[Any], request_id: UUID, message_id: str
) -> None:
    await conn.execute(
        "UPDATE approval_requests SET discord_message_id = %s WHERE id = %s",
        (message_id, request_id),
    )


async def get_request(
    conn: psycopg.AsyncConnection[Any], request_id: UUID
) -> ApprovalRequest | None:
    cur = await conn.execute(
        "SELECT id, session_id, turn, tool_use_id, tool_name, tool_input, status, requested_by,"
        " decided_by, self_approved, decided_at, expires_at, discord_message_id"
        " FROM approval_requests WHERE id = %s",
        (request_id,),
    )
    row = await cur.fetchone()
    return None if row is None else _request(row)


async def requests_for_tool_uses(
    conn: psycopg.AsyncConnection[Any], session_id: UUID, tool_use_ids: Sequence[str]
) -> dict[str, ApprovalRequest]:
    cur = await conn.execute(
        "SELECT id, session_id, turn, tool_use_id, tool_name, tool_input, status, requested_by,"
        " decided_by, self_approved, decided_at, expires_at, discord_message_id"
        " FROM approval_requests WHERE session_id = %s AND tool_use_id = ANY(%s)",
        (session_id, list(tool_use_ids)),
    )
    return {r.tool_use_id: r for r in map(_request, await cur.fetchall())}


async def decide_request(
    conn: psycopg.AsyncConnection[Any],
    request_id: UUID,
    status: Decided,
    decided_by: str | None,
) -> Decision:
    """Decide one pending request, once. A click on a request past `expires_at` expires it
    instead, exactly as the expiry pass would have."""
    async with conn.transaction():
        cur = await conn.execute(
            "SELECT session_id FROM approval_requests WHERE id = %s", (request_id,)
        )
        found = await cur.fetchone()
        if found is None:
            return Decision(changed=False, resume=False, request=None)
        session_id: UUID = found[0]
        # Serializes every decision for this session (and its resume trigger).
        await conn.execute("SELECT 1 FROM sessions WHERE id = %s FOR UPDATE", (session_id,))
        cur = await conn.execute(
            """
            UPDATE approval_requests
            SET status = CASE WHEN expires_at <= now() THEN 'expired' ELSE %(status)s::text END,
                decided_by = CASE WHEN expires_at <= now() OR %(status)s::text = 'expired'
                                  THEN NULL ELSE %(by)s::text END,
                self_approved = coalesce(expires_at > now() AND %(status)s::text = 'approved'
                                         AND %(by)s::text = requested_by, false),
                decided_at = now()
            WHERE id = %(id)s AND status = 'pending'
            RETURNING id, session_id, turn, tool_use_id, tool_name, tool_input, status,
                      requested_by, decided_by, self_approved, decided_at, expires_at,
                      discord_message_id
            """,
            {"status": status, "by": decided_by, "id": request_id},
        )
        row = await cur.fetchone()
        if row is None:
            return Decision(changed=False, resume=False, request=None)
        request = _request(row)

        cur = await conn.execute(
            "SELECT count(*) FROM approval_requests"
            " WHERE session_id = %s AND turn = %s AND status = 'pending'",
            (session_id, request.turn),
        )
        pending = await cur.fetchone()
        if pending is None or pending[0] > 0:
            return Decision(changed=True, resume=False, request=request)
        cur = await conn.execute(
            "UPDATE sessions SET status = 'running', heartbeat_at = now(), updated_at = now()"
            " WHERE id = %s AND status = 'awaiting_approval' RETURNING id",
            (session_id,),
        )
        flipped = await cur.fetchone()
        return Decision(changed=True, resume=flipped is not None, request=request)


async def expired_pending(conn: psycopg.AsyncConnection[Any]) -> list[UUID]:
    """Candidates for the expiry pass. Each still goes through decide_request."""
    cur = await conn.execute(
        "SELECT id FROM approval_requests WHERE status = 'pending' AND expires_at <= now()"
        " ORDER BY expires_at"
    )
    return [r[0] for r in await cur.fetchall()]


@dataclass(frozen=True)
class Receipt:
    tool_name: str
    tool_input: dict[str, object]
    outcome: Literal["succeeded", "failed"]
    result: dict[str, object]


async def receipts_for_session(
    conn: psycopg.AsyncConnection[Any], session_id: UUID
) -> list[Receipt]:
    cur = await conn.execute(
        "SELECT tool_name, tool_input, outcome, result FROM receipts"
        " WHERE session_id = %s ORDER BY executed_at, id",
        (session_id,),
    )
    return [Receipt(*r) for r in await cur.fetchall()]
