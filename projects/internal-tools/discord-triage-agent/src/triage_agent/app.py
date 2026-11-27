"""FastAPI app: POST /interactions (Discord) and GET /healthz, plus the startup check and
the sweep task.

Run locally: uvicorn triage_agent.app:app --port 8000
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import psycopg
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from triage_agent.agent.client import AnthropicModelClient, ModelClient
from triage_agent.agent.deps import Deps
from triage_agent.agent.pipeline import AskJob, handle_ask
from triage_agent.agent.reply import TokenStore, describe_error
from triage_agent.agent.resume import handle_click
from triage_agent.config import Settings
from triage_agent.db.sessions import NewSession, insert_session
from triage_agent.discord.components import parse_custom_id
from triage_agent.discord.rest import DiscordRest, HttpDiscordRest
from triage_agent.discord.verify import verify_signature
from triage_agent.sweep import SWEEP_INTERVAL_S, sweep_forever

log = logging.getLogger(__name__)

# Discord interaction and response types.
PING = 1
APPLICATION_COMMAND = 2
MESSAGE_COMPONENT = 3
PONG = 1
CHANNEL_MESSAGE = 4
DEFERRED_CHANNEL_MESSAGE = 5
DEFERRED_UPDATE_MESSAGE = 6
EPHEMERAL = 1 << 6

NOT_AUTHORIZED = "You're not authorized to approve or deny actions."

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


@dataclass(frozen=True)
class Clicker:
    user_id: str
    roles: tuple[str, ...]


def parse_clicker(payload: dict[str, Any]) -> Clicker | None:
    """The clicking member and their roles, from the interaction payload. None outside a
    guild (no member, so no roles)."""
    member = payload.get("member")
    if not isinstance(member, dict):
        return None
    user = member.get("user")
    user_id = user.get("id") if isinstance(user, dict) else None
    roles = member.get("roles")
    if not isinstance(user_id, str) or not isinstance(roles, list):
        return None
    return Clicker(user_id, tuple(r for r in roles if isinstance(r, str)))


def _deps(state: Any) -> Deps:
    cfg: Settings = state.settings
    return Deps(
        database_url=cfg.database_url,
        model=state.model,
        discord=state.discord,
        tokens=state.tokens,
        approvals=cfg.approvals,
    )


async def check_approver_channel(discord: DiscordRest, channel_id: str) -> None:
    """Startup check (SPEC.md section 4): refuse to start without the approvals channel."""
    try:
        await discord.get_channel(channel_id)
    except Exception as exc:
        raise RuntimeError(
            f"cannot fetch APPROVER_CHANNEL_ID {channel_id!r} with the bot token "
            f"({describe_error(exc)}); refusing to start"
        ) from exc


def create_app(
    settings: Settings | None = None,
    discord: DiscordRest | None = None,
    model: ModelClient | None = None,
    sweep_interval_s: float = SWEEP_INTERVAL_S,
) -> FastAPI:
    """Build the app. Tests pass settings and fake Discord and model clients; uvicorn passes
    none, and the app builds all three from the environment (.env) at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if not hasattr(app.state, "settings"):
            load_dotenv()
            app.state.settings = Settings.from_env()
        if not hasattr(app.state, "discord"):
            app.state.discord = HttpDiscordRest(app.state.settings.discord_bot_token)
        if not hasattr(app.state, "model"):
            app.state.model = AnthropicModelClient(app.state.settings.anthropic_api_key)
        await check_approver_channel(app.state.discord, app.state.settings.approvals.channel_id)
        # The first pass runs right away (the startup pass), in the background, so a resume
        # it starts never delays answering Discord.
        sweep = asyncio.create_task(sweep_forever(_deps(app.state), sweep_interval_s))
        try:
            yield
        finally:
            sweep.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sweep

    app = FastAPI(title="triage-agent", lifespan=lifespan)
    # Interaction tokens, in memory only (agent/reply.py).
    app.state.tokens = TokenStore()
    if settings is not None:
        app.state.settings = settings
    if discord is not None:
        app.state.discord = discord
    if model is not None:
        app.state.model = model

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
        if kind == MESSAGE_COMPONENT:
            return click(payload, cfg, request, background)
        if kind != APPLICATION_COMMAND:
            raise HTTPException(status_code=400, detail="unsupported interaction type")

        try:
            ask = parse_ask(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

        async with await psycopg.AsyncConnection.connect(cfg.database_url, autocommit=True) as conn:
            session_id, created = await insert_session(conn, ask.session)

        if created:
            # Everything past the ACK (model calls, the Discord reply) runs after the response.
            job = AskJob(
                session_id=session_id,
                question=ask.session.question,
                application_id=ask.application_id,
                token=ask.token,
            )
            background.add_task(
                handle_ask,
                cfg.database_url,
                request.app.state.model,
                request.app.state.discord,
                job,
                tokens=request.app.state.tokens,
                approvals=cfg.approvals,
            )
        return JSONResponse({"type": DEFERRED_CHANNEL_MESSAGE})

    def click(
        payload: dict[str, Any], cfg: Settings, request: Request, background: BackgroundTasks
    ) -> JSONResponse:
        """An Approve/Deny click. Only the role check and the ACK happen inline."""
        data = payload.get("data")
        try:
            parsed = parse_custom_id(data.get("custom_id") if isinstance(data, dict) else None)
        except ValueError:
            raise HTTPException(status_code=400, detail="unknown component") from None
        clicker = parse_clicker(payload)
        if clicker is None or cfg.approvals.role_id not in clicker.roles:
            return JSONResponse(
                {"type": CHANNEL_MESSAGE, "data": {"content": NOT_AUTHORIZED, "flags": EPHEMERAL}}
            )
        background.add_task(
            handle_click,
            _deps(request.app.state),
            parsed.request_id,
            parsed.action,
            clicker.user_id,
        )
        return JSONResponse({"type": DEFERRED_UPDATE_MESSAGE})

    return app


app = create_app()
