"""One /ask session after the ACK: classify, run the loop, then either send exactly one reply
or, if the loop paused, show the waiting note and post the approval requests."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID

import psycopg

from triage_agent.agent.approval_flow import apply_outcome
from triage_agent.agent.classifier import classify_session
from triage_agent.agent.client import ModelClient
from triage_agent.agent.deps import Deps
from triage_agent.agent.loop import fallback_reply, run_loop
from triage_agent.agent.reply import TokenStore, describe_error
from triage_agent.config import ApprovalConfig
from triage_agent.db.sessions import finish_session
from triage_agent.discord.rest import DiscordRest, fit_message

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AskJob:
    session_id: UUID
    question: str
    application_id: str
    token: str


async def _fallback(discord: DiscordRest, job: AskJob) -> str | None:
    """Edit the deferred response to the fallback. Returns the error to record, or None."""
    try:
        await discord.edit_original_response(
            job.application_id, job.token, fit_message(fallback_reply(job.session_id))
        )
    except Exception as exc:
        log.warning("fallback reply failed for session %s", job.session_id)
        return describe_error(exc, job.token)
    return None


async def handle_ask(
    database_url: str,
    model: ModelClient,
    discord: DiscordRest,
    job: AskJob,
    *,
    tokens: TokenStore | None = None,
    approvals: ApprovalConfig | None = None,
) -> None:
    """Background task. Never raises an Exception: an unexpected error before the outcome is
    handed off still ends in one fallback reply and a `failed` session."""
    tokens = tokens if tokens is not None else TokenStore()
    tokens.put(job.session_id, job.application_id, job.token)
    deps = Deps(
        database_url=database_url,
        model=model,
        discord=discord,
        tokens=tokens,
        approvals=approvals if approvals is not None else ApprovalConfig(),
    )
    handed_off = False
    try:
        async with await psycopg.AsyncConnection.connect(database_url, autocommit=True) as conn:
            classification = await classify_session(conn, model, job.session_id, job.question)
            outcome = await run_loop(
                conn,
                model,
                job.session_id,
                job.question,
                classification,
                approval_ttl=deps.approvals.ttl,
            )
            handed_off = True
            await apply_outcome(conn, deps, job.session_id, outcome)
        log.info("session %s %s (%s)", job.session_id, outcome.status, outcome.reason)
    except Exception:
        log.exception("session %s crashed", job.session_id)
        if not handed_off:
            reply_error = await _fallback(discord, job)
            tokens.drop(job.session_id)
            try:
                async with await psycopg.AsyncConnection.connect(
                    database_url, autocommit=True
                ) as conn:
                    await finish_session(conn, job.session_id, "failed", reply_error=reply_error)
            except Exception:
                log.exception("could not mark session %s failed", job.session_id)
