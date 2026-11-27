"""The PreToolUse gate's policy table (SPEC.md section 5). Plain code, deny by default.

The decision depends only on the tool name: nothing in the question, a ticket or the model's
output can change it.
"""

from __future__ import annotations

from typing import Final

from triage_agent.db.conversation import HookDecision

POLICY: Final[dict[str, HookDecision]] = {
    "get_ticket": "allow",
    "search_kb": "allow",
    "close_ticket": "require_approval",
}


def decide(tool_name: str) -> HookDecision:
    return POLICY.get(tool_name, "deny")
