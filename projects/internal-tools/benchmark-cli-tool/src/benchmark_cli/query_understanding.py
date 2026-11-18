"""Cheap rewrite of a question into search strings and an optional person filter.

The model does not answer. If the call fails, the raw question is the search.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from .llm import call_llm_structured
from .roster import person_by_label, roster_payload

logger = logging.getLogger(__name__)

_MAX_SEARCHES = 2
_MAX_SEARCH_CHARS = 500

_SYSTEM = """You rewrite the last message of a conversation into search strings for the
standalone question it asks. You do not answer it. Earlier turns are context: use them to
resolve what "that", "he", or "and what about Sam" refers to.
Return one or two short search strings. If the message names a roster person, including a
nickname, a possessive, or a typo, set person to that person's canonical name from the
roster and include it in a search string. The person is a search hint only. If the message
does not name a roster person, set person to null.
Use only canonical names from the roster. Do not invent people."""

_TURN_CHARS = 500
_TURNS = 4


class QueryPlan(BaseModel):
    """Search strings plus the roster person the follow-up names, if any."""

    searches: list[str] = Field(default_factory=list)
    person: str | None = None


def fallback_plan(
    question: str, thread_question: str | None = None, history: Sequence[Any] | None = None
) -> QueryPlan:
    """The raw question, with no person hint. The thread is not copied in."""
    del thread_question, history
    text = " ".join(question.split()) or question.strip()
    return QueryPlan(searches=[text] if text else [], person=None)


def _messages(
    question: str, thread_question: str | None, history: Sequence[Any] | None = None
) -> list[dict[str, Any]]:
    starter = " ".join((thread_question or "").split()) or "(none)"
    recent = [
        {"role": turn.role, "text": " ".join(turn.text.split())[:_TURN_CHARS]}
        for turn in list(history or ())[-_TURNS:]
    ]
    body: dict[str, Any] = {
        "thread_starter": starter,
        "recent_turns": recent,
        "question": question.strip(),
        "roster": roster_payload(),
    }
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": json.dumps(body, ensure_ascii=True)},
    ]


def _clean_searches(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    for item in values:
        text = " ".join(str(item).split())[:_MAX_SEARCH_CHARS]
        if text and text not in cleaned:
            cleaned.append(text)
        if len(cleaned) == _MAX_SEARCHES:
            break
    return cleaned


def understand_query(
    question: str, thread_question: str | None = None, history: Sequence[Any] | None = None
) -> QueryPlan:
    """Ask the cheap model for searches and a person. Failure uses the raw question."""
    if not question.strip():
        return fallback_plan(question, thread_question, history)
    try:
        parsed = call_llm_structured(
            tier="cheap",
            messages=_messages(question, thread_question, history),
            response_schema=QueryPlan,
            component="query_understanding",
        )
    except Exception:
        logger.info("Query understanding failed; searching the raw question.")
        return fallback_plan(question, thread_question, history)
    if not isinstance(parsed, QueryPlan):
        return fallback_plan(question, thread_question, history)
    searches = _clean_searches(parsed.searches)
    if not searches:
        return fallback_plan(question, thread_question, history)
    person = person_by_label(parsed.person)
    if parsed.person and person is None:
        logger.info("Query understanding returned a person who is not on the roster.")
    return QueryPlan(searches=searches, person=None if person is None else person.canonical)
