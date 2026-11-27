"""Discord REST egress over httpx."""

from __future__ import annotations

from typing import Any, Protocol

import httpx

DISCORD_API = "https://discord.com/api/v10"
# Every Discord call is capped well under the 2 min limit (SPEC.md section 2).
REQUEST_TIMEOUT_S = 30.0
# Discord rejects message content over 2000 characters.
MAX_MESSAGE_CHARS = 2000
# Replies carry model text, which is untrusted: never let it ping @everyone, roles or users.
NO_MENTIONS: dict[str, list[str]] = {"parse": []}


def fit_message(content: str) -> str:
    """Cut content to Discord's limit, marking the cut with an ellipsis."""
    if len(content) <= MAX_MESSAGE_CHARS:
        return content
    return content[: MAX_MESSAGE_CHARS - 1] + "…"


class DiscordRest(Protocol):
    async def edit_original_response(
        self, application_id: str, interaction_token: str, content: str
    ) -> None: ...

    async def get_channel(self, channel_id: str) -> dict[str, Any]: ...

    async def post_channel_message(
        self,
        channel_id: str,
        content: str,
        *,
        components: list[dict[str, Any]] | None = None,
        allowed_mentions: dict[str, Any] | None = None,
    ) -> str:
        """Post as the bot. Returns the new message's id."""
        ...

    async def edit_channel_message(
        self,
        channel_id: str,
        message_id: str,
        content: str,
        *,
        components: list[dict[str, Any]],
    ) -> None: ...


class HttpDiscordRest:
    def __init__(self, bot_token: str | None, client: httpx.AsyncClient | None = None) -> None:
        self._bot_token = bot_token
        self._client = client

    async def edit_original_response(
        self, application_id: str, interaction_token: str, content: str
    ) -> None:
        # Interaction webhooks are authorized by the token in the URL; no bot token.
        url = f"{DISCORD_API}/webhooks/{application_id}/{interaction_token}/messages/@original"
        if self._client is not None:
            await self._patch(self._client, url, content)
            return
        async with httpx.AsyncClient() as client:
            await self._patch(client, url, content)

    @staticmethod
    async def _patch(client: httpx.AsyncClient, url: str, content: str) -> None:
        body = {"content": fit_message(content), "allowed_mentions": NO_MENTIONS}
        resp = await client.patch(url, json=body, timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()

    # --- bot-token REST (approvals channel, replies after 15 min) -------------------------

    async def _bot(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        if not self._bot_token:
            raise RuntimeError("DISCORD_BOT_TOKEN is not set")
        headers = {"Authorization": f"Bot {self._bot_token}"}
        url = f"{DISCORD_API}{path}"
        if self._client is not None:
            resp = await self._client.request(
                method, url, json=body, headers=headers, timeout=REQUEST_TIMEOUT_S
            )
        else:
            async with httpx.AsyncClient() as client:
                resp = await client.request(
                    method, url, json=body, headers=headers, timeout=REQUEST_TIMEOUT_S
                )
        resp.raise_for_status()
        return resp.json() if resp.content else None

    async def get_channel(self, channel_id: str) -> dict[str, Any]:
        data = await self._bot("GET", f"/channels/{channel_id}")
        if not isinstance(data, dict):
            raise RuntimeError(f"unexpected channel response for {channel_id}")
        return data

    async def post_channel_message(
        self,
        channel_id: str,
        content: str,
        *,
        components: list[dict[str, Any]] | None = None,
        allowed_mentions: dict[str, Any] | None = None,
    ) -> str:
        body: dict[str, Any] = {
            "content": fit_message(content),
            "allowed_mentions": allowed_mentions if allowed_mentions is not None else NO_MENTIONS,
        }
        if components is not None:
            body["components"] = components
        data = await self._bot("POST", f"/channels/{channel_id}/messages", body)
        if not isinstance(data, dict) or not isinstance(data.get("id"), str):
            raise RuntimeError("Discord returned no message id")
        message_id: str = data["id"]
        return message_id

    async def edit_channel_message(
        self,
        channel_id: str,
        message_id: str,
        content: str,
        *,
        components: list[dict[str, Any]],
    ) -> None:
        body = {
            "content": fit_message(content),
            "components": components,
            "allowed_mentions": NO_MENTIONS,
        }
        await self._bot("PATCH", f"/channels/{channel_id}/messages/{message_id}", body)
