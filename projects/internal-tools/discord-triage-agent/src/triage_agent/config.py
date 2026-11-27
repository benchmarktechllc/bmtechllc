"""Settings read from the environment (a local .env in v1, fly secrets later)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Final

DEFAULT_APPROVAL_TTL: Final = timedelta(hours=24)
_TTL_UNITS: Final = {"s": 1, "m": 60, "h": 3600}


class MissingSettingError(RuntimeError):
    pass


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise MissingSettingError(f"{name} is not set (see .env.example)")
    return value


def parse_ttl(text: str) -> timedelta:
    """Parse an APPROVAL_TTL like `24h`, `5m`, `90s` or `300` (seconds). Must be positive."""
    match = re.fullmatch(r"(\d+)([smh]?)", text.strip().lower())
    if match is None or int(match.group(1)) == 0:
        raise ValueError(f"APPROVAL_TTL must look like 24h, 5m, 90s or 300, not {text!r}")
    return timedelta(seconds=int(match.group(1)) * _TTL_UNITS[match.group(2) or "s"])


@dataclass(frozen=True)
class ApprovalConfig:
    """Where approvals go and who may decide them (SPEC.md section 1). Changing these
    means a restart. The empty defaults only keep code paths that never pause short; the
    app requires both ids (`Settings.from_env`) and checks the channel at startup."""

    channel_id: str = ""
    role_id: str = ""
    ttl: timedelta = DEFAULT_APPROVAL_TTL


@dataclass(frozen=True)
class Settings:
    discord_public_key: str
    database_url: str
    discord_bot_token: str | None = None
    anthropic_api_key: str | None = None
    approvals: ApprovalConfig = field(default_factory=ApprovalConfig)

    @classmethod
    def from_env(cls) -> Settings:
        ttl_text = os.environ.get("APPROVAL_TTL", "").strip()
        return cls(
            discord_public_key=require_env("DISCORD_PUBLIC_KEY"),
            database_url=require_env("DATABASE_URL"),
            discord_bot_token=os.environ.get("DISCORD_BOT_TOKEN") or None,
            anthropic_api_key=require_env("ANTHROPIC_API_KEY"),
            approvals=ApprovalConfig(
                channel_id=require_env("APPROVER_CHANNEL_ID"),
                role_id=require_env("APPROVER_ROLE_ID"),
                ttl=parse_ttl(ttl_text) if ttl_text else DEFAULT_APPROVAL_TTL,
            ),
        )
