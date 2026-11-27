"""Approve/Deny buttons and the text of approval messages (SPEC.md section 5).

Ticket subjects and tool inputs are untrusted text. They only ever appear inside inline code
(backticks removed, so they cannot close it), and approval messages go out with no mentions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Literal
from uuid import UUID

ACTION_ROW: Final = 1
BUTTON: Final = 2
SUCCESS: Final = 3
DANGER: Final = 4

Action = Literal["approve", "deny"]
_CUSTOM_ID = re.compile(r"approval:([0-9a-fA-F-]{36}):(approve|deny)")
_INLINE_MAX = 300


@dataclass(frozen=True)
class Click:
    request_id: UUID
    action: Action


def custom_id(request_id: UUID, action: Action) -> str:
    return f"approval:{request_id}:{action}"


def parse_custom_id(value: object) -> Click:
    """Raises ValueError unless `value` is `approval:<uuid>:approve|deny`."""
    match = _CUSTOM_ID.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError("unknown component")
    action: Action = "approve" if match.group(2) == "approve" else "deny"
    return Click(UUID(match.group(1)), action)


def approval_buttons(request_id: UUID) -> list[dict[str, Any]]:
    return [
        {
            "type": ACTION_ROW,
            "components": [
                {
                    "type": BUTTON,
                    "style": SUCCESS,
                    "label": "Approve",
                    "custom_id": custom_id(request_id, "approve"),
                },
                {
                    "type": BUTTON,
                    "style": DANGER,
                    "label": "Deny",
                    "custom_id": custom_id(request_id, "deny"),
                },
            ],
        }
    ]


def inline(text: str) -> str:
    """Show untrusted text as inline code that it cannot break out of."""
    cleaned = " ".join(text.replace("`", "'").split())
    if len(cleaned) > _INLINE_MAX:
        cleaned = cleaned[: _INLINE_MAX - 1] + "…"
    return f"`{cleaned or ' '}`"


@dataclass(frozen=True)
class TicketSummary:
    id: str
    subject: str
    status: str


def request_text(
    *,
    session_id: UUID,
    asker_user_id: str,
    tool_name: str,
    tool_input: object,
    ticket: TicketSummary | None,
    expires_at: datetime,
) -> str:
    if ticket is None:
        ticket_line = "Ticket: not found"
    else:
        ticket_line = (
            f"Ticket: {inline(ticket.id)} {inline(ticket.subject)} (status {inline(ticket.status)})"
        )
    return "\n".join(
        [
            f"**Approval needed:** {inline(tool_name)} (session {session_id})",
            f"Asked by <@{asker_user_id}>",
            ticket_line,
            f"Input: {inline(json.dumps(tool_input, ensure_ascii=False, sort_keys=True))}",
            f"Expires {_when(expires_at)}",
        ]
    )


def decided_text(
    original: str,
    status: Literal["approved", "denied", "expired"],
    decided_by: str | None,
    decided_at: datetime,
) -> str:
    if status == "expired":
        outcome = f"⌛ **Expired** at {_when(decided_at)}, with no decision"
    else:
        label = "✅ **Approved**" if status == "approved" else "❌ **Denied**"
        who = f"<@{decided_by}>" if decided_by else "unknown"
        outcome = f"{label} by {who} at {_when(decided_at)}"
    return f"{original}\n\n{outcome}"


def _when(at: datetime) -> str:
    return at.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
