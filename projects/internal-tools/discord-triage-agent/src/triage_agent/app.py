"""FastAPI app: POST /interactions (Discord) and GET /healthz.

Run locally: uvicorn triage_agent.app:app --port 8000
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import psycopg
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from triage_agent.config import Settings
from triage_agent.db.sessions import NewSession, insert_session
from triage_agent.discord.rest import DiscordRest, HttpDiscordRest
from triage_agent.discord.verify import verify_signature

log = logging.getLogger(__name__)

# Discord interaction and response types.
PING = 1
APPLICATION_COMMAND = 2
PONG = 1
DEFERRED_CHANNEL_MESSAGE = 5

ASK_COMMAND_NAME = "ask"


@dataclass(frozen=True)
class AskInteraction:
    application_id: str
    token: str
    session: NewSession


def parse_ask(payload: dict[str, Any]) -> AskInteraction:
    """Pull the fields /ask needs out of an interaction. Raises ValueError if any are missing."""
    data = payload.get("data")
    if not isinstance(data, dict) or data.get("name") != ASK_COMMAND_NAME:
        raise ValueError("unknown command")
    question = next(
        (
            o.get("value")
            for o in data.get("options") or []
            if isinstance(o, dict) and o.get("name") == "question"
        ),
        None,
    )
    member = payload.get("member")
    user = member.get("user") if isinstance(member, dict) else payload.get("user")
    asker = user.get("id") if isinstance(user, dict) else None
    channel = payload.get("channel")
    channel_id = payload.get("channel_id") or (
        channel.get("id") if isinstance(channel, dict) else None
    )
    fields = {
        "interaction id": payload.get("id"),
        "application id": payload.get("application_id"),
        "token": payload.get("token"),
        "guild id": payload.get("guild_id"),
        "channel id": channel_id,
        "asker": asker,
        "question": question,
    }
    missing = [name for name, value in fields.items() if not isinstance(value, str) or not value]
    if missing:
        raise ValueError(f"missing {', '.join(missing)}")
    return AskInteraction(
        application_id=str(payload["application_id"]),
        token=str(payload["token"]),
        session=NewSession(
            interaction_id=str(payload["id"]),
            guild_id=str(payload["guild_id"]),
            channel_id=str(channel_id),
            asker_user_id=str(asker),
            question=str(question),
        ),
    )


def received_message(session_id: UUID) -> str:
    return f"Received — session {session_id}"


async def send_received(
    discord: DiscordRest, application_id: str, token: str, session_id: UUID
) -> None:
    """Background task: edit the deferred response. Runs after the ACK has been sent."""
    try:
        await discord.edit_original_response(application_id, token, received_message(session_id))
    except Exception:
        log.exception("failed to edit original response for session %s", session_id)


def create_app(settings: Settings | None = None, discord: DiscordRest | None = None) -> FastAPI:
    """Build the app. Tests pass settings and a fake Discord client; uvicorn passes neither,
    and the app loads both from the environment (.env) at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if not hasattr(app.state, "settings"):
            load_dotenv()
            app.state.settings = Settings.from_env()
        if not hasattr(app.state, "discord"):
            app.state.discord = HttpDiscordRest(app.state.settings.discord_bot_token)
        yield

    app = FastAPI(title="triage-agent", lifespan=lifespan)
    if settings is not None:
        app.state.settings = settings
    if discord is not None:
        app.state.discord = discord

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/interactions")
    async def interactions(request: Request, background: BackgroundTasks) -> JSONResponse:
        cfg: Settings = request.app.state.settings
        body = await request.body()
        if not verify_signature(
            cfg.discord_public_key,
            request.headers.get("X-Signature-Ed25519"),
            request.headers.get("X-Signature-Timestamp"),
            body,
        ):
            raise HTTPException(status_code=401, detail="invalid request signature")

        try:
            payload = json.loads(body)
        except ValueError:
            raise HTTPException(status_code=400, detail="body is not JSON") from None
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="body is not a JSON object")

        kind = payload.get("type")
        if kind == PING:
            return JSONResponse({"type": PONG})
        if kind != APPLICATION_COMMAND:
            raise HTTPException(status_code=400, detail="unsupported interaction type")

        try:
            ask = parse_ask(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

        async with await psycopg.AsyncConnection.connect(cfg.database_url, autocommit=True) as conn:
            session_id, created = await insert_session(conn, ask.session)

        if created:
            discord_client: DiscordRest = request.app.state.discord
            background.add_task(
                send_received, discord_client, ask.application_id, ask.token, session_id
            )
        return JSONResponse({"type": DEFERRED_CHANNEL_MESSAGE})

    return app


app = create_app()
