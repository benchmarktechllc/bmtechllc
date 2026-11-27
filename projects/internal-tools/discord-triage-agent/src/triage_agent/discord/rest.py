"""Discord REST egress over httpx."""

from __future__ import annotations

from typing import Protocol

import httpx

DISCORD_API = "https://discord.com/api/v10"
# Every Discord call is capped well under the 2 min limit (SPEC.md section 2).
REQUEST_TIMEOUT_S = 30.0


class DiscordRest(Protocol):
    async def edit_original_response(
        self, application_id: str, interaction_token: str, content: str
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
        resp = await client.patch(url, json={"content": content}, timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
