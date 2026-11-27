"""One /ask session after the ACK: classify, run the loop, send exactly one reply."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID

import psycopg

from triage_agent.agent.classifier import classify_session
from triage_agent.agent.client import ModelClient
from triage_agent.agent.loop import fallback_reply, run_loop
from triage_agent.db.sessions import finish_session
from triage_agent.discord.rest import DiscordRest, fit_message

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AskJob:
    session_id: UUID
    question: str
    application_id: str
    token: str


async def _reply(discord: DiscordRest, job: AskJob, content: str) -> str | None:
    """Edit the deferred response. Returns the error to record, or None on success."""
    try:
        await discord.edit_original_response(job.application_id, job.token, fit_message(content))
    except Exception as exc:
        log.warning("reply failed for session %s", job.session_id)
        # httpx errors include the request URL, which contains the interaction token.
        return f"{type(exc).__name__}: {exc}".replace(job.token, "<token>")[:500]
    return None


async def handle_ask(
    database_url: str, model: ModelClient, discord: DiscordRest, job: AskJob
) -> None:
    """Background task. Never raises: an unexpected error still ends in one fallback reply and
    a `failed` session, unless a reply already went out."""
    replied = False
    try:
        async with await psycopg.AsyncConnection.connect(database_url, autocommit=True) as conn:
            classification = await classify_session(conn, model, job.session_id, job.question)
            outcome = await run_loop(conn, model, job.session_id, job.question, classification)
            reply_error = await _reply(discord, job, outcome.reply)
            replied = True
            await finish_session(conn, job.session_id, outcome.status, reply_error=reply_error)
        log.info("session %s %s (%s)", job.session_id, outcome.status, outcome.reason)
    except Exception:
        log.exception("session %s crashed", job.session_id)
        if not replied:
            reply_error = await _reply(discord, job, fallback_reply(job.session_id))
            try:
                async with await psycopg.AsyncConnection.connect(
                    database_url, autocommit=True
                ) as conn:
                    await finish_session(conn, job.session_id, "failed", reply_error=reply_error)
            except Exception:
                log.exception("could not mark session %s failed", job.session_id)
