"""Reply routing (SPEC.md section 2, step 4). Plain code.

Within 15 min of the interaction the reply edits the deferred original response; after that
(or after a restart, since tokens live only in memory) it goes out by bot-token REST to the
asker's channel. A failed REST post records `reply_error`, still marks the session finished,
and posts a notice in the approvals channel that mentions the asker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Final
from uuid import UUID

import psycopg

from triage_agent.db.sessions import SessionStatus, finish_session
from triage_agent.discord.rest import NO_MENTIONS, DiscordRest, fit_message

log = logging.getLogger(__name__)

# Interaction tokens last 15 min; stop using one a minute early.
TOKEN_WINDOW_MINUTES: Final = 14


@dataclass(frozen=True)
class InteractionToken:
    application_id: str
    token: str


class TokenStore:
    """Interaction tokens by session, in memory only: a credential is never written to the
    database. After a restart the store is empty and replies go out by bot-token REST."""

    def __init__(self) -> None:
        self._tokens: dict[UUID, InteractionToken] = {}

    def put(self, session_id: UUID, application_id: str, token: str) -> None:
        self._tokens[session_id] = InteractionToken(application_id, token)

    def get(self, session_id: UUID) -> InteractionToken | None:
        return self._tokens.get(session_id)

    def drop(self, session_id: UUID) -> None:
        self._tokens.pop(session_id, None)


def describe_error(exc: BaseException, secret: str | None = None) -> str:
    """An error for the database or the log. httpx errors include the request URL, which
    holds the interaction token for webhook calls, so the token is removed."""
    text = f"{type(exc).__name__}: {exc}"
    if secret:
        text = text.replace(secret, "<token>")
    return text[:500]


@dataclass(frozen=True)
class _Target:
    channel_id: str
    asker_user_id: str
    token: InteractionToken | None


async def _target(
    conn: psycopg.AsyncConnection[Any], tokens: TokenStore, session_id: UUID
) -> _Target:
    cur = await conn.execute(
        "SELECT channel_id, asker_user_id,"
        " now() - created_at < make_interval(mins => %s) FROM sessions WHERE id = %s",
        (TOKEN_WINDOW_MINUTES, session_id),
    )
    row = await cur.fetchone()
    if row is None:
        raise LookupError(f"session {session_id} not found")
    channel_id, asker, within_window = row
    return _Target(channel_id, asker, tokens.get(session_id) if within_window else None)


async def send_waiting_note(
    conn: psycopg.AsyncConnection[Any],
    discord: DiscordRest,
    tokens: TokenStore,
    session_id: UUID,
    note: str,
) -> None:
    """Show a status note in the deferred reply, if its token is still usable. It is not a
    reply: when the token is gone there is nothing to edit, and the asker hears from the bot
    once, with the answer."""
    target = await _target(conn, tokens, session_id)
    if target.token is None:
        return
    try:
        await discord.edit_original_response(
            target.token.application_id, target.token.token, fit_message(note)
        )
    except Exception as exc:
        log.warning(
            "waiting note failed for session %s: %s",
            session_id,
            describe_error(exc, target.token.token),
        )


async def deliver_final(
    conn: psycopg.AsyncConnection[Any],
    discord: DiscordRest,
    tokens: TokenStore,
    approver_channel_id: str,
    session_id: UUID,
    content: str,
    status: SessionStatus,
) -> None:
    """Send the session's one reply and finish it with `status`."""
    target = await _target(conn, tokens, session_id)
    reply_error: str | None = None
    if target.token is not None:
        try:
            await discord.edit_original_response(
                target.token.application_id, target.token.token, fit_message(content)
            )
        except Exception as exc:
            log.warning("reply edit failed for session %s", session_id)
            reply_error = describe_error(exc, target.token.token)
    else:
        try:
            await discord.post_channel_message(
                target.channel_id, fit_message(content), allowed_mentions=NO_MENTIONS
            )
        except Exception as exc:
            log.warning("reply post failed for session %s", session_id)
            reply_error = describe_error(exc)
            await _notify_approvers(discord, approver_channel_id, target.asker_user_id, session_id)
    await finish_session(conn, session_id, status, reply_error=reply_error)
    tokens.drop(session_id)


async def _notify_approvers(
    discord: DiscordRest, approver_channel_id: str, asker_user_id: str, session_id: UUID
) -> None:
    try:
        await discord.post_channel_message(
            approver_channel_id,
            f"<@{asker_user_id}> I couldn't post the answer to your question in its channel "
            f"(session {session_id}). Please ask a teammate to check it.",
            allowed_mentions={"parse": [], "users": [asker_user_id]},
        )
    except Exception as exc:
        log.error("notice for session %s failed too: %s", session_id, describe_error(exc))
