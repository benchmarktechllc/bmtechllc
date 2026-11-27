"""Register /ask for the test guild (bulk overwrite of that guild's commands).

Usage: python scripts/register_commands.py
Needs DISCORD_APPLICATION_ID, DISCORD_GUILD_ID and DISCORD_BOT_TOKEN in .env.
"""

from __future__ import annotations

from dotenv import load_dotenv

from triage_agent.config import require_env
from triage_agent.discord.commands import register_guild_commands


def main() -> None:
    load_dotenv()
    commands = register_guild_commands(
        application_id=require_env("DISCORD_APPLICATION_ID"),
        guild_id=require_env("DISCORD_GUILD_ID"),
        bot_token=require_env("DISCORD_BOT_TOKEN"),
    )
    for cmd in commands:
        print(f"registered /{cmd.get('name')} (id {cmd.get('id')})")


if __name__ == "__main__":
    main()
