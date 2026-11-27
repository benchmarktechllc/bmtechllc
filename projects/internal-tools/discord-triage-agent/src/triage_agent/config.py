"""Settings read from the environment (a local .env in v1, fly secrets later)."""

from __future__ import annotations

import os
from dataclasses import dataclass


class MissingSettingError(RuntimeError):
    pass


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise MissingSettingError(f"{name} is not set (see .env.example)")
    return value


@dataclass(frozen=True)
class Settings:
    discord_public_key: str
    database_url: str
    discord_bot_token: str | None = None

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            discord_public_key=require_env("DISCORD_PUBLIC_KEY"),
            database_url=require_env("DATABASE_URL"),
            discord_bot_token=os.environ.get("DISCORD_BOT_TOKEN") or None,
        )
