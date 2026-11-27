"""sessions table access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb


@dataclass(frozen=True)
class NewSession:
    interaction_id: str
    guild_id: str
    channel_id: str
    asker_user_id: str
    question: str


async def insert_session(conn: psycopg.AsyncConnection[Any], s: NewSession) -> tuple[UUID, bool]:
    """Insert a session for this interaction, once. Returns (session_id, created).

    A repeated interaction_id returns the existing row's id with created=False.
    """
    cur = await conn.execute(
        """
        INSERT INTO sessions (interaction_id, guild_id, channel_id, asker_user_id, question)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (interaction_id) DO NOTHING
        RETURNING id
        """,
        (s.interaction_id, s.guild_id, s.channel_id, s.asker_user_id, s.question),
    )
    row = await cur.fetchone()
    if row is not None:
        return row[0], True
    cur = await conn.execute(
        "SELECT id FROM sessions WHERE interaction_id = %s", (s.interaction_id,)
    )
    existing = await cur.fetchone()
    if existing is None:  # pragma: no cover - the conflicting row cannot vanish mid-call
        raise RuntimeError(f"session for interaction {s.interaction_id} disappeared")
    return existing[0], False


SessionStatus = Literal["running", "awaiting_approval", "completed", "failed"]


async def store_classification(
    conn: psycopg.AsyncConnection[Any],
    session_id: UUID,
    *,
    concerns: list[dict[str, str]],
    more_concerns: bool,
    primary_category: str,
    categories: list[str],
    confidence: float | None,
) -> None:
    """Store the (validated) classifier result on the session and bump its heartbeat."""
    await conn.execute(
        """
        UPDATE sessions
        SET concerns = %s, more_concerns = %s, primary_category = %s, categories = %s,
            confidence = %s, heartbeat_at = now(), updated_at = now()
        WHERE id = %s
        """,
        (Jsonb(concerns), more_concerns, primary_category, categories, confidence, session_id),
    )


async def touch_heartbeat(
    conn: psycopg.AsyncConnection[Any], session_id: UUID, *, iterations: int | None = None
) -> None:
    """Set heartbeat_at to now, and the loop's iteration count when given."""
    await conn.execute(
        """
        UPDATE sessions
        SET heartbeat_at = now(), updated_at = now(), iterations = coalesce(%s, iterations)
        WHERE id = %s
        """,
        (iterations, session_id),
    )


async def finish_session(
    conn: psycopg.AsyncConnection[Any],
    session_id: UUID,
    status: SessionStatus,
    *,
    reply_error: str | None = None,
) -> None:
    await conn.execute(
        """
        UPDATE sessions
        SET status = %s, reply_error = %s, heartbeat_at = now(), updated_at = now()
        WHERE id = %s
        """,
        (status, reply_error, session_id),
    )


async def set_status(
    conn: psycopg.AsyncConnection[Any], session_id: UUID, status: SessionStatus
) -> None:
    """Change the status (and heartbeat) without touching reply_error."""
    await conn.execute(
        "UPDATE sessions SET status = %s, heartbeat_at = now(), updated_at = now() WHERE id = %s",
        (status, session_id),
    )
