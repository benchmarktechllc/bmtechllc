"""Discord REST egress and guild command registration, via httpx.MockTransport (no network)."""

from __future__ import annotations

import json

import httpx
import pytest

from triage_agent.discord.commands import ASK_COMMAND, register_guild_commands
from triage_agent.discord.rest import DISCORD_API, REQUEST_TIMEOUT_S, HttpDiscordRest


def test_request_timeout_is_at_most_two_minutes() -> None:
    assert 0 < REQUEST_TIMEOUT_S <= 120


@pytest.mark.anyio
async def test_edit_original_response_patches_the_webhook_message() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "m1"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        rest = HttpDiscordRest(bot_token="bot-secret", client=http)
        await rest.edit_original_response("app-1", "tok-1", "Received — session abc")

    assert len(seen) == 1
    req = seen[0]
    assert req.method == "PATCH"
    assert str(req.url) == f"{DISCORD_API}/webhooks/app-1/tok-1/messages/@original"
    assert json.loads(req.content) == {"content": "Received — session abc"}
    # Interaction webhooks are authorized by the token in the URL, not the bot token.
    assert "authorization" not in req.headers
    assert req.extensions["timeout"]["read"] == REQUEST_TIMEOUT_S


@pytest.mark.anyio
async def test_edit_original_response_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "Unknown Webhook"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        rest = HttpDiscordRest(bot_token=None, client=http)
        with pytest.raises(httpx.HTTPStatusError):
            await rest.edit_original_response("app-1", "tok-1", "x")


def test_ask_command_definition() -> None:
    assert ASK_COMMAND["name"] == "ask"
    assert ASK_COMMAND["type"] == 1
    (option,) = ASK_COMMAND["options"]
    assert option == {
        "name": "question",
        "description": option["description"],
        "type": 3,
        "required": True,
    }


def test_register_guild_commands_bulk_overwrites_guild_commands() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=[{"id": "cmd-1", **ASK_COMMAND}])

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        result = register_guild_commands("app-1", "guild-1", "bot-secret", client=http)

    assert len(seen) == 1
    req = seen[0]
    assert req.method == "PUT"
    assert str(req.url) == f"{DISCORD_API}/applications/app-1/guilds/guild-1/commands"
    assert req.headers["authorization"] == "Bot bot-secret"
    assert json.loads(req.content) == [ASK_COMMAND]
    assert req.extensions["timeout"]["read"] == REQUEST_TIMEOUT_S
    assert result[0]["id"] == "cmd-1"


def test_register_guild_commands_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "401: Unauthorized"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(httpx.HTTPStatusError):
            register_guild_commands("app-1", "guild-1", "bad", client=http)
