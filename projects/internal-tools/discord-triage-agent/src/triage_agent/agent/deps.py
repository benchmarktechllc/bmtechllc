"""What the background work (the /ask pipeline, clicks, resume, the sweep) needs to run."""

from __future__ import annotations

from dataclasses import dataclass

from triage_agent.agent.client import ModelClient
from triage_agent.agent.reply import TokenStore
from triage_agent.config import ApprovalConfig
from triage_agent.discord.rest import DiscordRest


@dataclass(frozen=True)
class Deps:
    database_url: str
    model: ModelClient
    discord: DiscordRest
    tokens: TokenStore
    approvals: ApprovalConfig
