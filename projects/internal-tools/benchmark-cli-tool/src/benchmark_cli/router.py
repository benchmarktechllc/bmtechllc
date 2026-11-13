"""Chat model selection for the answer layer.

Every question uses the configured cheap chat model. There is no flagship
escalation and no second call on a larger model.
"""

from __future__ import annotations


def classify_query(question: str) -> str:
    """Return the only chat tier. The question is accepted so callers stay stable."""
    if not question.strip():
        raise ValueError("question must not be empty.")
    return "cheap"
