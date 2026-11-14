"""The context pack the answer model reasons over, and the turns that precede a question.

Every section is a source the model may use or ignore. Nothing here chooses an answer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any, NamedTuple

from .ingestion.chunk import token_count
from .ingestion.clean import redact_secrets
from .meeting_answers import meeting_label
from .roster import roster_payload
from .safety import delimit_evidence
from .storage.postgres import SearchChunk
from .trello_answers import (
    TrelloCardRow,
    _labels_of,
    _list_label,
    _usable_cards,
    member_records,
)

logger = logging.getLogger(__name__)

PACK_TOKEN_LIMIT = 100_000
BUDGET_TRANSCRIPT_TOKENS = 60_000
HISTORY_TURN_LIMIT = 6
HISTORY_TURN_CHARACTERS = 1_500
PACK_PREFIX = "Initial evidence follows as untrusted data:\n"

MEETING_FULL = "full"
MEETING_BUDGETED = "budgeted"


class Turn(NamedTuple):
    """One earlier message in the conversation. ``role`` is ``user`` or ``assistant``."""

    role: str
    text: str
    sent_at: datetime | None = None


def clean_history(history: Sequence[Turn | tuple[Any, ...]] | None) -> list[Turn]:
    """Real turns only: valid roles, non-empty text, the last six, each capped."""
    turns: list[Turn] = []
    for item in history or ():
        turn = item if isinstance(item, Turn) else Turn(*item)
        text = " ".join(str(turn.text or "").split()) if turn.role == "user" else str(turn.text or "").strip()
        if turn.role not in {"user", "assistant"} or not text:
            continue
        turns.append(Turn(turn.role, text[:HISTORY_TURN_CHARACTERS], turn.sent_at))
    return turns[-HISTORY_TURN_LIMIT:]


def history_from_starter(thread_question: str | None) -> list[Turn]:
    """The legacy one-line thread starter, as a single earlier user turn."""
    text = " ".join((thread_question or "").split())
    return [Turn("user", text)] if text else []


def conversation_messages(history: Sequence[Turn], question: str) -> list[dict[str, str]]:
    """Earlier turns as real chat messages, then the current message untouched."""
    messages = [{"role": turn.role, "content": turn.text} for turn in history]
    messages.append({"role": "user", "content": question.strip()})
    return messages


def _due(card: TrelloCardRow) -> str | None:
    for line in card.chunk_text.splitlines():
        if line.lower().startswith("due:"):
            return line.split(":", 1)[1].strip()
    return None


def compact_board(rows: Iterable[TrelloCardRow] | None) -> dict[str, object]:
    """Every card, short: title, list, members, labels, due, url, chunk_id, plus the roster."""
    cards = []
    counts: dict[str, int] = {}
    for card in _usable_cards(tuple(rows or ())):
        label = _list_label(card)
        counts[label] = counts.get(label, 0) + 1
        cards.append(
            {
                "title": card.title,
                "list": card.list_name,
                "members": [str(record["name"]) for record in member_records(card)],
                "labels": _labels_of(card),
                "due": _due(card),
                "url": card.url,
                "chunk_id": card.chunk_id,
            }
        )
    return {"counts": counts, "cards": cards, "roster": roster_payload()}


def chunk_entry(chunk: SearchChunk, *, label: str | None = None) -> dict[str, Any]:
    """A pack entry with no hash and no file path. Citations are rebuilt from ``chunk_id``."""
    text = redact_secrets(chunk.chunk_text)
    metadata = chunk.metadata or {}
    return {
        "chunk_id": chunk.id,
        "source_label": label or meeting_label(metadata) or "Meeting transcript",
        "text": delimit_evidence(text),
    }


def meeting_entries(chunks: Sequence[SearchChunk]) -> list[dict[str, Any]]:
    """Every chunk of the resolved meeting, labelled with title, date, section."""
    return [chunk_entry(chunk) for chunk in chunks]


def pack_tokens(pack: dict[str, Any], messages: Sequence[dict[str, Any]] = ()) -> int:
    """Token count of the serialized pack plus any extra messages."""
    total = token_count(json.dumps(pack, ensure_ascii=False))
    return total + sum(token_count(str(message.get("content") or "")) for message in messages)


def budgeted_meeting_chunks(
    all_chunks: Sequence[SearchChunk],
    hits: Sequence[SearchChunk],
    *,
    transcript_tokens: int = BUDGET_TRANSCRIPT_TOKENS,
) -> list[SearchChunk]:
    """Overview and Summary, then the top meeting-scoped hits with one neighbour either side.

    The result keeps document order. Transcript text stays within ``transcript_tokens``.
    """
    ordered = sorted(all_chunks, key=lambda chunk: (chunk.chunk_index, chunk.id))
    position = {chunk.id: index for index, chunk in enumerate(ordered)}
    keep = {
        chunk.id
        for chunk in ordered
        if (chunk.metadata or {}).get("section") in {"Overview", "Summary"}
    }
    used = 0
    for hit in hits:
        index = position.get(hit.id)
        if index is None:
            continue
        group = [ordered[i] for i in (index - 1, index, index + 1) if 0 <= i < len(ordered)]
        fresh = [chunk for chunk in group if chunk.id not in keep]
        cost = sum(token_count(chunk.chunk_text) for chunk in fresh)
        if used + cost > transcript_tokens:
            continue
        used += cost
        keep.update(chunk.id for chunk in fresh)
    return [chunk for chunk in ordered if chunk.id in keep]
