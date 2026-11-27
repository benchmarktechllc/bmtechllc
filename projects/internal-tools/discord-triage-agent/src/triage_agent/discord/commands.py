"""Slash command definitions and guild registration."""

from __future__ import annotations

from typing import Any

import httpx

from triage_agent.discord.rest import DISCORD_API, REQUEST_TIMEOUT_S

CHAT_INPUT = 1
STRING_OPTION = 3

ASK_COMMAND: dict[str, Any] = {
    "name": "ask",
    "type": CHAT_INPUT,
    "description": "Ask a support question about a ticket or how-to topic",
    "options": [
        {
            "name": "question",
            "description": "Your question",
            "type": STRING_OPTION,
            "required": True,
        }
    ],
}


def register_guild_commands(
    application_id: str,
    guild_id: str,
    bot_token: str,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    """Bulk-overwrite this guild's commands with [/ask]. Returns Discord's response."""
    url = f"{DISCORD_API}/applications/{application_id}/guilds/{guild_id}/commands"
    headers = {"Authorization": f"Bot {bot_token}"}
    if client is None:
        with httpx.Client() as own:
            resp = own.put(url, json=[ASK_COMMAND], headers=headers, timeout=REQUEST_TIMEOUT_S)
    else:
        resp = client.put(url, json=[ASK_COMMAND], headers=headers, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    result: list[dict[str, Any]] = resp.json()
    return result
