"""The in-app sweep (SPEC.md section 4): every 5 min, and once at startup so anything that went
stale while the app was stopped is handled right away.

- Expiry pass: expires `pending` requests past `expires_at` through the same decision
  transaction as a click, and resumes a session once its turn has no pending request.
- Recovery pass: a session is stale only when it is `running` with a heartbeat older than
  10 min. It is claimed with a conditional UPDATE, so two passes can't take the same session,
  then either finished (a fully decided paused turn) or failed with the apology.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import Any, Final
from uuid import UUID

import psycopg

from triage_agent.agent.approval_flow import show_decision
from triage_agent.agent.deps import Deps
from triage_agent.agent.resume import fail_with_apology, paused_turn, resume_or_apologize
from triage_agent.db.approvals import decide_request, expired_pending

log = logging.getLogger(__name__)

SWEEP_INTERVAL_S: Final = 300
STALE_AFTER: Final = timedelta(minutes=10)


async def expiry_pass(conn: psycopg.AsyncConnection[Any], deps: Deps) -> None:
    for request_id in await expired_pending(conn):
        try:
            decision = await decide_request(conn, request_id, "expired", None)
            if not decision.changed or decision.request is None:
                continue  # a click decided it first
            await show_decision(conn, deps, decision.request)
            if decision.resume:
                await resume_or_apologize(conn, deps, decision.request.session_id)
        except Exception:
            log.exception("expiry of approval %s failed", request_id)


async def _claim_stale(conn: psycopg.AsyncConnection[Any]) -> list[UUID]:
    # Bumping the heartbeat is the claim: a concurrent pass no longer sees these as stale.
    cur = await conn.execute(
        "UPDATE sessions SET heartbeat_at = now(), updated_at = now()"
        " WHERE status = 'running' AND heartbeat_at < now() - %s RETURNING id",
        (STALE_AFTER,),
    )
    return [r[0] for r in await cur.fetchall()]


async def recovery_pass(conn: psycopg.AsyncConnection[Any], deps: Deps) -> None:
    for session_id in await _claim_stale(conn):
        try:
            if await paused_turn(conn, session_id) is not None:
                log.info("recovering session %s by resuming it", session_id)
                await resume_or_apologize(conn, deps, session_id)
            else:
                log.info("recovering session %s: marking it failed", session_id)
                await fail_with_apology(conn, deps, session_id)
        except Exception:
            log.exception("recovery of session %s failed", session_id)


async def run_sweep_once(deps: Deps) -> None:
    async with await psycopg.AsyncConnection.connect(deps.database_url, autocommit=True) as conn:
        await expiry_pass(conn, deps)
        await recovery_pass(conn, deps)


async def sweep_forever(deps: Deps, interval_s: float = SWEEP_INTERVAL_S) -> None:
    """The first pass runs at once (the startup pass), then one every `interval_s`."""
    while True:
        try:
            await run_sweep_once(deps)
        except Exception:
            log.exception("sweep failed")
        await asyncio.sleep(interval_s)
