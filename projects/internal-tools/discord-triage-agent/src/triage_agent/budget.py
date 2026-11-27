"""Model prices and the live-run budget guard (SPEC.md section 8). Plain arithmetic.

Used by scripts/live_smoke.py: it adds up each response's cost and, before each session,
refuses to start one whose worst case could push the total past BUDGET_LIMIT_USD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from triage_agent.agent.classifier import CLASSIFIER_MAX_TOKENS, CLASSIFIER_MODEL, MAX_CONCERNS
from triage_agent.agent.loop import AGENT_MAX_TOKENS, AGENT_MODEL, MAX_ITERATIONS


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float


# Standard first-party API prices in USD per million tokens (Haiku: prompts up to 100K).
PRICES_CHECKED: Final = "2026-11-03"
PRICES: Final[dict[str, Price]] = {
    "claude-sonnet-5-5": Price(input_per_mtok=2.00, output_per_mtok=10.00),
    "claude-haiku-5-5": Price(input_per_mtok=0.10, output_per_mtok=0.50),
}

BUDGET_LIMIT_USD: Final = 4.0

# Conservative input sizes for the ceiling. Output is capped by max_tokens; input is not, so
# these are deliberate overestimates (system + tools + full history + tool results).
AGENT_INPUT_TOKENS_CEILING: Final = 20_000
CLASSIFIER_INPUT_TOKENS_CEILING: Final = 4_000
GRADER_MODEL: Final = CLASSIFIER_MODEL
GRADER_MAX_TOKENS: Final = 256
GRADER_INPUT_TOKENS_CEILING: Final = 4_000
# Fixtures list at most this many expected concerns; each is graded by its own call.
MAX_GRADED_CONCERNS: Final = MAX_CONCERNS + 1


def call_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD for one call. Raises KeyError for a model with no recorded price."""
    price = PRICES[model]
    return (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1_000_000


def session_ceiling_usd() -> float:
    """Worst case for one live session: the classifier, 8 agent calls at the caps, graders."""
    classifier = call_cost(CLASSIFIER_MODEL, CLASSIFIER_INPUT_TOKENS_CEILING, CLASSIFIER_MAX_TOKENS)
    agent = MAX_ITERATIONS * call_cost(AGENT_MODEL, AGENT_INPUT_TOKENS_CEILING, AGENT_MAX_TOKENS)
    graders = MAX_GRADED_CONCERNS * call_cost(
        GRADER_MODEL, GRADER_INPUT_TOKENS_CEILING, GRADER_MAX_TOKENS
    )
    return classifier + agent + graders


def can_start_session(
    spent_usd: float, ceiling_usd: float | None = None, limit_usd: float = BUDGET_LIMIT_USD
) -> bool:
    ceiling = session_ceiling_usd() if ceiling_usd is None else ceiling_usd
    # Rounded so that spent == limit - ceiling isn't refused over a float rounding error.
    return round(spent_usd + ceiling, 9) <= limit_usd
