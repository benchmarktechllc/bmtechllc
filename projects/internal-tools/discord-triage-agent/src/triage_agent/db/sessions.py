"""sessions table access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import psycopg


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
