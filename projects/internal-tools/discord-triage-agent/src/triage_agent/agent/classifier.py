"""Classify a question into 1-3 concerns with one structured-output call (SPEC.md section 2).

The model proposes; plain code validates. Anything invalid, or a confidence below the cutoff,
becomes a single `other` concern whose summary is the question itself.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Final
from uuid import UUID

import anthropic
import psycopg
from anthropic.types import Message

from triage_agent.agent.client import ModelClient, ModelRequest
from triage_agent.agent.prompts import CLASSIFIER_SYSTEM
from triage_agent.db.conversation import Trace, insert_trace
from triage_agent.db.sessions import store_classification

CLASSIFIER_MODEL: Final = "claude-haiku-5-5"
CLASSIFIER_MAX_TOKENS: Final = 256
CONFIDENCE_CUTOFF: Final = 0.6
MAX_CONCERNS: Final = 3
MAX_SUMMARY_CHARS: Final = 300
CATEGORIES: Final = ("billing", "account", "bug", "how_to", "other")
CLASSIFIER_ITERATION: Final = 0  # traces.iteration for the classifier call

OUTPUT_SCHEMA: Final[dict[str, object]] = {
    "type": "object",
    "properties": {
        "concerns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                },
                "required": ["summary", "category"],
                "additionalProperties": False,
            },
        },
        "primary": {"type": "integer"},
        "confidence": {"type": "number"},
        "more_concerns": {"type": "boolean"},
    },
    "required": ["concerns", "primary", "confidence", "more_concerns"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Concern:
    summary: str
    category: str


@dataclass(frozen=True)
class Classification:
    concerns: tuple[Concern, ...]
    more_concerns: bool
    primary_category: str
    categories: tuple[str, ...]
    confidence: float | None


def fallback_classification(question: str, confidence: float | None = None) -> Classification:
    return Classification(
        concerns=(Concern(summary=question, category="other"),),
        more_concerns=False,
        primary_category="other",
        categories=("other",),
        confidence=confidence,
    )


def _distinct(items: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def parse_classification(question: str, text: str) -> Classification:
    """Validate the classifier's JSON text. Pure: no I/O."""
    try:
        data = json.loads(text)
    except ValueError:
        return fallback_classification(question)
    if not isinstance(data, dict):
        return fallback_classification(question)

    raw_concerns = data.get("concerns")
    primary = data.get("primary")
    confidence = data.get("confidence")
    more = data.get("more_concerns")
    if (
        not isinstance(raw_concerns, list)
        or not raw_concerns
        or not isinstance(primary, int)
        or isinstance(primary, bool)
        or not _is_number(confidence)
        or not isinstance(more, bool)
    ):
        return fallback_classification(question)
    assert isinstance(confidence, int | float)  # narrowed by _is_number above
    if not 0.0 <= confidence <= 1.0:
        return fallback_classification(question)

    concerns: list[Concern] = []
    for item in raw_concerns:
        if not isinstance(item, dict):
            return fallback_classification(question)
        summary, category = item.get("summary"), item.get("category")
        if not isinstance(summary, str) or not summary.strip() or category not in CATEGORIES:
            return fallback_classification(question)
        assert isinstance(category, str)
        concerns.append(Concern(summary=summary.strip()[:MAX_SUMMARY_CHARS], category=category))

    more_concerns = more or len(concerns) > MAX_CONCERNS
    concerns = concerns[:MAX_CONCERNS]
    if not 0 <= primary < len(concerns):
        return fallback_classification(question)
    if confidence < CONFIDENCE_CUTOFF:
        return fallback_classification(question, confidence=float(confidence))

    return Classification(
        concerns=tuple(concerns),
        more_concerns=more_concerns,
        primary_category=concerns[primary].category,
        categories=_distinct([c.category for c in concerns]),
        confidence=float(confidence),
    )


def classifier_request(question: str) -> ModelRequest:
    return ModelRequest(
        model=CLASSIFIER_MODEL,
        max_tokens=CLASSIFIER_MAX_TOKENS,
        system=CLASSIFIER_SYSTEM,
        messages=[{"role": "user", "content": question}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        # Haiku 5.5 thinks by default; at 256 tokens thinking could eat the whole answer.
        # Disabling it is allowed at effort high or below.
        thinking={"type": "disabled"},
    )


def response_text(response: Message) -> str:
    return "".join(b.text for b in response.content if b.type == "text")


async def classify_session(
    conn: psycopg.AsyncConnection[Any], model: ModelClient, session_id: UUID, question: str
) -> Classification:
    """Classify, trace the call (no messages row: that table is the agent conversation),
    and store the result on the session. Never raises on a model-side failure."""
    start = time.monotonic()
    try:
        response = await model.create(classifier_request(question))
    except anthropic.APIError as exc:
        await insert_trace(
            conn,
            Trace(
                session_id=session_id,
                iteration=CLASSIFIER_ITERATION,
                model=CLASSIFIER_MODEL,
                latency_ms=_ms_since(start),
                error=f"{type(exc).__name__}: {exc}"[:500],
            ),
        )
        result = fallback_classification(question)
    else:
        text = response_text(response) if response.stop_reason == "end_turn" else ""
        result = parse_classification(question, text)
        await insert_trace(
            conn,
            Trace(
                session_id=session_id,
                iteration=CLASSIFIER_ITERATION,
                model=response.model,
                stop_reason=response.stop_reason,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                tool_result={"classifier_output": response_text(response)},
                latency_ms=_ms_since(start),
            ),
        )

    await store_classification(
        conn,
        session_id,
        concerns=[{"summary": c.summary, "category": c.category} for c in result.concerns],
        more_concerns=result.more_concerns,
        primary_category=result.primary_category,
        categories=list(result.categories),
        confidence=result.confidence,
    )
    return result


def _ms_since(start: float) -> int:
    return int((time.monotonic() - start) * 1000)
