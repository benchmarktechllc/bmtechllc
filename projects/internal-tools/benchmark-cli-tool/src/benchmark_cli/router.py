"""Chat model selection for the answer layer.

Helper and tool-loop calls use the configured cheap chat model. Answer
synthesis uses its own "synthesis" tier. There is no flagship escalation.
"""

from __future__ import annotations


def classify_query(question: str) -> str:
    """Return the only chat tier. The question is accepted so callers stay stable."""
    if not question.strip():
        raise ValueError("question must not be empty.")
    return "cheap"
