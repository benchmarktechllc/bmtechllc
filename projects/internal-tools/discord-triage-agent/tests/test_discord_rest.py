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
    # Model text is untrusted: it must never ping @everyone, roles or users.
    assert json.loads(req.content) == {
        "content": "Received — session abc",
        "allowed_mentions": {"parse": []},
    }
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


@pytest.mark.anyio
async def test_edit_original_response_caps_content_at_2000_characters() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "m1"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        await HttpDiscordRest(bot_token=None, client=http).edit_original_response(
            "app-1", "tok-1", "y" * 2500
        )
    content = json.loads(seen[0].content)["content"]
    assert len(content) == 2000
    assert content.endswith("…")


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


# --- bot-token REST (Slice 3) --------------------------------------------------------


@pytest.mark.anyio
async def test_bot_calls_use_the_bot_token_and_the_timeout() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"id": "c1", "type": 0})
        if request.method == "POST":
            return httpx.Response(200, json={"id": "m42"})
        return httpx.Response(200, json={"id": "m42"})

    components = [{"type": 1, "components": []}]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        rest = HttpDiscordRest(bot_token="bot-secret", client=http)
        assert (await rest.get_channel("c1"))["id"] == "c1"
        message_id = await rest.post_channel_message("c1", "z" * 2500, components=components)
        await rest.edit_channel_message("c1", "m42", "decided", components=[])

    assert message_id == "m42"
    get, post, patch = seen
    assert (get.method, str(get.url)) == ("GET", f"{DISCORD_API}/channels/c1")
    assert (post.method, str(post.url)) == ("POST", f"{DISCORD_API}/channels/c1/messages")
    assert (patch.method, str(patch.url)) == ("PATCH", f"{DISCORD_API}/channels/c1/messages/m42")
    for req in seen:
        assert req.headers["authorization"] == "Bot bot-secret"
        assert req.extensions["timeout"]["read"] == REQUEST_TIMEOUT_S
    posted = json.loads(post.content)
    assert len(posted["content"]) == 2000
    assert posted["components"] == components
    assert posted["allowed_mentions"] == {"parse": []}
    assert json.loads(patch.content) == {
        "content": "decided",
        "components": [],
        "allowed_mentions": {"parse": []},
    }


@pytest.mark.anyio
async def test_post_channel_message_passes_explicit_mentions_and_omits_components() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "m1"})

    mentions = {"parse": [], "users": ["u1"]}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        await HttpDiscordRest("bot", client=http).post_channel_message(
            "c1", "hi <@u1>", allowed_mentions=mentions
        )
    body = json.loads(seen[0].content)
    assert body == {"content": "hi <@u1>", "allowed_mentions": mentions}


@pytest.mark.anyio
async def test_bot_calls_raise_on_http_error_and_without_a_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"message": "Missing Access", "code": 50001})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(httpx.HTTPStatusError):
            await HttpDiscordRest("bot", client=http).get_channel("c1")
        with pytest.raises(RuntimeError, match="DISCORD_BOT_TOKEN"):
            await HttpDiscordRest(None, client=http).get_channel("c1")
