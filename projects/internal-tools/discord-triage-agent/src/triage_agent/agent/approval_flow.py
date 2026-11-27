"""What happens after the loop returns, for /ask and for a resume alike: either post the
approval requests of a paused turn, or send the session's one reply. Plus the edit that shows
a decision on its approval message."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

import psycopg

from triage_agent.agent.deps import Deps
from triage_agent.agent.loop import LoopOutcome
from triage_agent.agent.reply import deliver_final, describe_error, send_waiting_note
from triage_agent.db.approvals import ApprovalRequest, get_request, set_message_id
from triage_agent.discord.components import (
    TicketSummary,
    approval_buttons,
    decided_text,
    request_text,
)
from triage_agent.discord.rest import NO_MENTIONS

log = logging.getLogger(__name__)


async def apply_outcome(
    conn: psycopg.AsyncConnection[Any], deps: Deps, session_id: UUID, outcome: LoopOutcome
) -> None:
    if outcome.status == "awaiting_approval":
        # The note goes first: buttons can only be clicked once posted, so a fast decision's
        # final reply can never be overwritten by a late note.
        await send_waiting_note(conn, deps.discord, deps.tokens, session_id, outcome.reply)
        for request_id in outcome.pending:
            await post_approval_message(conn, deps, request_id)
        return
    await deliver_final(
        conn,
        deps.discord,
        deps.tokens,
        deps.approvals.channel_id,
        session_id,
        outcome.reply,
        outcome.status,
    )


async def _request_text(conn: psycopg.AsyncConnection[Any], request: ApprovalRequest) -> str:
    cur = await conn.execute(
        "SELECT asker_user_id FROM sessions WHERE id = %s", (request.session_id,)
    )
    session = await cur.fetchone()
    raw_id = request.tool_input.get("ticket_id")
    ticket: TicketSummary | None = None
    if isinstance(raw_id, str):
        cur = await conn.execute(
            "SELECT id, subject, status FROM tickets WHERE id = %s", (raw_id.strip().upper(),)
        )
        row = await cur.fetchone()
        ticket = None if row is None else TicketSummary(*row)
    return request_text(
        session_id=request.session_id,
        asker_user_id=session[0] if session else "unknown",
        tool_name=request.tool_name,
        tool_input=request.tool_input,
        ticket=ticket,
        expires_at=request.expires_at,
    )


async def post_approval_message(
    conn: psycopg.AsyncConnection[Any], deps: Deps, request_id: UUID
) -> None:
    """Post one Approve/Deny message and save its id. A failure is logged: the request stays
    pending and expires after the TTL, which resumes the session with "approval expired"."""
    request = await get_request(conn, request_id)
    if request is None:
        return
    try:
        message_id = await deps.discord.post_channel_message(
            deps.approvals.channel_id,
            await _request_text(conn, request),
            components=approval_buttons(request.id),
            allowed_mentions=NO_MENTIONS,
        )
    except Exception as exc:
        log.error("could not post approval %s: %s", request_id, describe_error(exc))
        return
    await set_message_id(conn, request_id, message_id)


async def show_decision(
    conn: psycopg.AsyncConnection[Any], deps: Deps, request: ApprovalRequest
) -> None:
    """Edit the approval message to show the outcome, who decided and when, with no buttons."""
    if request.discord_message_id is None or request.status == "pending":
        return
    if request.decided_at is None:  # pragma: no cover - the CHECK constraint forbids it
        return
    content = decided_text(
        await _request_text(conn, request), request.status, request.decided_by, request.decided_at
    )
    try:
        await deps.discord.edit_channel_message(
            deps.approvals.channel_id, request.discord_message_id, content, components=[]
        )
    except Exception as exc:
        log.warning("could not update approval %s: %s", request.id, describe_error(exc))
