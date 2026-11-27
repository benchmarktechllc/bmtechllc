"""Resume a paused session (SPEC.md section 5, Resume), and the click that triggers it.

The same session continues, never a new one. Resume reloads the stored messages, finds the
paused turn (the latest assistant message, with approval requests for its tool calls), and
appends one user message holding a tool_result for every tool_use_id of that turn, in order,
unless that message is already stored. Every result is rebuilt from stored data (traces,
decisions, receipts), and the close executor is idempotent, so running the resume again (as
recovery does) never closes a ticket twice or reuses a seq.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

import psycopg

from triage_agent.agent.approval_flow import apply_outcome, show_decision
from triage_agent.agent.classifier import Classification, Concern, fallback_classification
from triage_agent.agent.deps import Deps
from triage_agent.agent.loop import continue_loop, tool_result_block
from triage_agent.agent.reply import deliver_final
from triage_agent.db.approvals import (
    ApprovalRequest,
    Receipt,
    decide_request,
    receipts_for_session,
    requests_for_tool_uses,
)
from triage_agent.db.conversation import JsonObject, append_message, load_messages
from triage_agent.db.sessions import touch_heartbeat
from triage_agent.tools.close_ticket import execute_close
from triage_agent.tools.executors import ToolOutcome

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PausedTurn:
    tool_use_ids: tuple[str, ...]
    requests: dict[str, ApprovalRequest]
    results_stored: bool


async def paused_turn(conn: psycopg.AsyncConnection[Any], session_id: UUID) -> PausedTurn | None:
    """The session's paused turn if its latest assistant message is one and every request of
    it is decided; otherwise None (nothing to resume)."""
    messages = await load_messages(conn, session_id)
    last = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i].role == "assistant"), None
    )
    if last is None:
        return None
    ids = tuple(
        str(b["id"]) for b in messages[last].content if b.get("type") == "tool_use" and "id" in b
    )
    if not ids:
        return None
    requests = await requests_for_tool_uses(conn, session_id, ids)
    if not requests or any(r.status == "pending" for r in requests.values()):
        return None
    return PausedTurn(ids, requests, results_stored=last + 1 < len(messages))


async def _approver_name(conn: psycopg.AsyncConnection[Any], user_id: str | None) -> str:
    if user_id is None:
        return "unknown"
    cur = await conn.execute("SELECT name FROM teammates WHERE discord_user_id = %s", (user_id,))
    row = await cur.fetchone()
    return str(row[0]) if row else user_id


async def _saved_result(
    conn: psycopg.AsyncConnection[Any], session_id: UUID, tool_use_id: str
) -> ToolOutcome:
    cur = await conn.execute(
        "SELECT tool_result FROM traces WHERE session_id = %s AND tool_use_id = %s"
        " AND tool_result IS NOT NULL ORDER BY id DESC LIMIT 1",
        (session_id, tool_use_id),
    )
    row = await cur.fetchone()
    saved = row[0] if row else None
    if not isinstance(saved, dict) or not isinstance(saved.get("content"), str):
        return ToolOutcome("tool result unavailable", is_error=True)
    return ToolOutcome(saved["content"], is_error=bool(saved.get("is_error")))


async def _result_for(
    conn: psycopg.AsyncConnection[Any],
    session_id: UUID,
    tool_use_id: str,
    request: ApprovalRequest | None,
) -> ToolOutcome:
    if request is None:  # an allowed or denied call: its result was saved by the gate
        return await _saved_result(conn, session_id, tool_use_id)
    if request.status == "approved":
        return await execute_close(conn, request.id)
    if request.status == "denied":
        return ToolOutcome(
            f"denied by {await _approver_name(conn, request.decided_by)}", is_error=True
        )
    return ToolOutcome("approval expired", is_error=True)


def _classification(row: Sequence[Any]) -> Classification:
    question, concerns, more_concerns, primary_category, categories, confidence = row
    parsed = tuple(
        Concern(summary=str(c["summary"]), category=str(c["category"]))
        for c in concerns or []
        if isinstance(c, dict) and "summary" in c and "category" in c
    )
    if not parsed:
        return fallback_classification(question)
    return Classification(
        concerns=parsed,
        more_concerns=bool(more_concerns),
        primary_category=primary_category or parsed[0].category,
        categories=tuple(categories or ()),
        confidence=confidence,
    )


async def resume_session(conn: psycopg.AsyncConnection[Any], deps: Deps, session_id: UUID) -> bool:
    """Resume a `running` session from its paused turn. Returns False, changing nothing, if
    the session is not running or has no fully decided paused turn."""
    cur = await conn.execute(
        "SELECT status, iterations, question, concerns, more_concerns, primary_category,"
        " categories, confidence FROM sessions WHERE id = %s",
        (session_id,),
    )
    row = await cur.fetchone()
    if row is None or row[0] != "running":
        return False
    turn = await paused_turn(conn, session_id)
    if turn is None:
        return False

    if not turn.results_stored:
        results: list[JsonObject] = []
        for tool_use_id in turn.tool_use_ids:
            result = await _result_for(
                conn, session_id, tool_use_id, turn.requests.get(tool_use_id)
            )
            results.append(tool_result_block(tool_use_id, result))
        async with conn.transaction():
            await append_message(conn, session_id, "user", results)
            await touch_heartbeat(conn, session_id)

    outcome = await continue_loop(
        conn,
        deps.model,
        session_id,
        _classification(row[2:]),
        start_iteration=row[1] + 1,
        approval_ttl=deps.approvals.ttl,
    )
    await apply_outcome(conn, deps, session_id, outcome)
    log.info("session %s resumed: %s (%s)", session_id, outcome.status, outcome.reason)
    return True


async def resume_or_apologize(
    conn: psycopg.AsyncConnection[Any], deps: Deps, session_id: UUID
) -> None:
    """Resume; if that raises, fail the session with the apology rather than leave it to
    recovery. (A crash of the process itself is what recovery is for.)"""
    try:
        await resume_session(conn, deps, session_id)
    except Exception:
        log.exception("resume of session %s failed", session_id)
        await fail_with_apology(conn, deps, session_id)


async def handle_click(
    deps: Deps, request_id: UUID, action: Literal["approve", "deny"], user_id: str
) -> None:
    """Background task for a role holder's click (already ACKed with type 6). Never raises."""
    try:
        async with await psycopg.AsyncConnection.connect(
            deps.database_url, autocommit=True
        ) as conn:
            status: Literal["approved", "denied"] = "approved" if action == "approve" else "denied"
            decision = await decide_request(conn, request_id, status, user_id)
            if not decision.changed or decision.request is None:
                return  # unknown or already decided: a no-op
            await show_decision(conn, deps, decision.request)
            if decision.resume:
                await resume_or_apologize(conn, deps, decision.request.session_id)
    except Exception:
        log.exception("click on approval %s failed", request_id)


# --- the apology (recovery, or a resume that raised) ----------------------------------


def _ticket_label(raw: object) -> str:
    # Model-written input: keep only ticket-id characters, so the label is plain text.
    return re.sub(r"[^A-Za-z0-9-]", "", str(raw or "")).upper()[:20] or "a ticket"


def _receipt_line(receipt: Receipt) -> str:
    ticket = _ticket_label(receipt.tool_input.get("ticket_id"))
    if receipt.outcome == "failed":
        return f"closing {ticket} failed"
    try:
        already = json.loads(str(receipt.result.get("content"))).get("already_closed") is True
    except (ValueError, AttributeError):
        already = False
    return f"{ticket} was already closed" if already else f"{ticket} was closed"


def apology_text(session_id: UUID, receipts: Sequence[Receipt]) -> str:
    text = f"Sorry, I couldn't finish answering this question (session {session_id})."
    if not receipts:
        return f"{text} Nothing was changed."
    return f"{text} Already done: {'; '.join(_receipt_line(r) for r in receipts)}."


async def fail_with_apology(
    conn: psycopg.AsyncConnection[Any], deps: Deps, session_id: UUID
) -> None:
    """Mark the session failed and send the fixed apology, listing every receipt."""
    receipts = await receipts_for_session(conn, session_id)
    await deliver_final(
        conn,
        deps.discord,
        deps.tokens,
        deps.approvals.channel_id,
        session_id,
        apology_text(session_id, receipts),
        "failed",
    )
