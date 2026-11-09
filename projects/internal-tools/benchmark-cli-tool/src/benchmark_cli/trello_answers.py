"""Answer Trello list-membership questions from indexed cards.

Hybrid retrieval returns only a few chunks, so it cannot enumerate a board
list. A narrow membership question is answered from the first indexed chunk
of each card. Setup, history, and other Trello questions stay on the normal
grounded path.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .ingestion.clean import redact_secrets
from .models import AnswerResponse, Citation

MAX_LISTED_CARDS = 25
_HASH = re.compile(r"^[0-9a-f]{64}$")
_PROGRESS = frozenset({"doing", "in progress", "active in progress"})
_NOT_MEMBERSHIP = re.compile(
    r"\b(?:how|why|set[\s-]?up|setup|configure|config|integrat\w*|connector|"
    r"api|sync\w*|ingest\w*|tokens?|webhooks?|work(?:ing|s)?)\b",
    re.IGNORECASE,
)
_LIST_TAIL = re.compile(
    r"""(?ix)
    (?:\bin\b|\bunder\b)\s+
    (?:the\s+)?
    (?P<name>
        ["“'‘][^"”'’]{1,40}["”'’]
        |
        [^\n?.!]{1,60}?
    )
    (?:\s+(?:column|list|lane|state|status|stage))?
    \s*[?.!]*\s*$
    """
)


@dataclass(frozen=True, slots=True)
class TrelloCardRow:
    """First indexed chunk of one Trello card."""

    board: str
    list_name: str
    card_id: str
    title: str
    url: str
    chunk_id: int
    source_file: str
    content_sha256: str
    chunk_text: str


def card_row_from_tuple(value: object) -> TrelloCardRow | None:
    """Cast one ``trello_card_rows`` tuple. A malformed row is omitted."""
    if isinstance(value, TrelloCardRow):
        return value
    if not isinstance(value, tuple) or len(value) != 9:
        return None
    board, list_name, card_id, title, url, chunk_id, source_file, digest, chunk_text = value
    if isinstance(chunk_id, bool):
        return None
    try:
        parsed_id = int(chunk_id)
    except (TypeError, ValueError):
        return None
    if parsed_id < 1:
        return None
    return TrelloCardRow(
        board=str(board or "").strip(),
        list_name=str(list_name or "").strip(),
        card_id=str(card_id or "").strip(),
        title=str(title or "").strip(),
        url=str(url or "").strip(),
        chunk_id=parsed_id,
        source_file=str(source_file or "").strip(),
        content_sha256=str(digest or "").strip().lower(),
        chunk_text=str(chunk_text or ""),
    )


def cards_from_rows(rows: Iterable[object]) -> tuple[TrelloCardRow, ...]:
    """Keep every tuple that casts to a card row."""
    parsed: list[TrelloCardRow] = []
    for item in rows:
        card = card_row_from_tuple(item)
        if card is not None and card.card_id:
            parsed.append(card)
    return tuple(parsed)


def requested_trello_list(question: str) -> str | None:
    """Return a normalized list name for a narrow membership question.

    The question must mention a card, ticket, or Trello, ask which/what/list/show,
    and end with a list name after ``in`` or ``under``. ``in progress`` and
    ``active in progress`` are the Doing list. Procedure and history questions
    return ``None`` so ordinary retrieval still answers them.
    """
    text = " ".join(question.split())
    if not text or _NOT_MEMBERSHIP.search(text):
        return None
    folded = text.casefold()
    if re.search(r"\b(?:cards?|tickets?|trello)\b", folded) is None:
        return None
    if re.search(r"\b(?:which|what|list|show)\b", folded) is None:
        return None
    match = _LIST_TAIL.search(text)
    if match is None:
        return None
    normalized = _normalize(match.group("name").strip("\"'“”‘’"))
    if not normalized or len(normalized) > 40:
        return None
    if normalized in _PROGRESS:
        return "doing"
    return normalized


def trello_list_answer(requested: str, rows: Sequence[TrelloCardRow]) -> AnswerResponse:
    """List indexed cards in one Trello list, without descriptions or comments."""
    usable = _usable_cards(rows)
    if not usable:
        return AnswerResponse(
            status="no_relevant_information",
            answer="No Trello cards are in the retrieval index yet.",
            confidence=0.0,
            citations=[],
        )
    matched = [card for card in usable if _list_matches(card.list_name, requested)]
    if not matched:
        names = ", ".join(_indexed_list_names(usable))
        label = requested.strip()
        return AnswerResponse(
            status="no_relevant_information",
            answer=(
                f'No indexed Trello cards are in a list named "{label}". '
                f"Lists with indexed cards: {names}."
            ),
            confidence=0.0,
            citations=[],
        )
    shown = matched[:MAX_LISTED_CARDS]
    hidden = len(matched) - len(shown)
    lines = _render_groups(shown)
    if hidden:
        lines.append(f"…and {hidden} more.")
    return AnswerResponse(
        status="answered",
        answer="\n".join(lines),
        confidence=0.8,
        citations=[_citation(card) for card in shown],
    )


def _normalize(value: str) -> str:
    text = value.casefold().replace("-", " ")
    text = re.sub(r"[\"'“”‘’]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _list_matches(indexed: str, requested: str) -> bool:
    """Match a list name, a phrase-boundary suffix, or the Doing alias."""
    left = _normalize(indexed)
    right = _normalize(requested)
    if not left or not right:
        return False
    if left == right or left.endswith(" " + right):
        return True
    return _is_progress(left) and _is_progress(right)


def _is_progress(name: str) -> bool:
    return name in _PROGRESS or any(name.endswith(" " + alias) for alias in _PROGRESS)


def _usable_cards(rows: Sequence[TrelloCardRow]) -> list[TrelloCardRow]:
    kept: list[TrelloCardRow] = []
    seen: set[tuple[str, str]] = set()
    for card in rows:
        key = (card.board, card.card_id)
        quote = _quote(card)
        if (
            not card.card_id
            or not card.list_name
            or not card.source_file
            or card.chunk_id < 1
            or key in seen
            or _HASH.fullmatch(card.content_sha256) is None
            or not quote
        ):
            continue
        seen.add(key)
        kept.append(card)
    return kept


def _indexed_list_names(cards: Sequence[TrelloCardRow]) -> list[str]:
    seen: dict[str, str] = {}
    for card in cards:
        label = " ".join(redact_secrets(card.list_name).split()) or "Untitled list"
        key = _normalize(label)
        if key and key not in seen:
            seen[key] = label
    return [seen[key] for key in sorted(seen)]


def _quote(card: TrelloCardRow) -> str:
    for line in card.chunk_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Card: "):
            redacted = redact_secrets(stripped).strip()
            if redacted:
                return redacted
    return " ".join(redact_secrets(card.title).split())


def _citation(card: TrelloCardRow) -> Citation:
    return Citation(
        source_file=card.source_file,
        chunk_id=card.chunk_id,
        content_sha256=card.content_sha256,
        quote=_quote(card),
    )


def _render_groups(cards: Sequence[TrelloCardRow]) -> list[str]:
    groups: dict[str, list[TrelloCardRow]] = {}
    for card in cards:
        label = " ".join(redact_secrets(card.list_name).split()) or "Untitled list"
        groups.setdefault(label, []).append(card)
    lines: list[str] = []
    for index, (name, members) in enumerate(groups.items()):
        if index:
            lines.append("")
        lines.append(f"Trello cards in {name}:")
        lines.extend(_bullet(card) for card in members)
    return lines


def _bullet(card: TrelloCardRow) -> str:
    title = " ".join(redact_secrets(card.title).split()) or "Untitled card"
    if card.url.startswith("https://"):
        return f"- [{title}]({card.url})"
    return f"- {title}"
