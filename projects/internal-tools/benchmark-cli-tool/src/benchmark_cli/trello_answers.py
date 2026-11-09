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
from .storage.postgres import SearchChunk

MAX_LISTED_CARDS = 25
OPEN_LISTS = "__open_lists__"
_DISCORD_MESSAGE_LIMIT = 2000
# Compact source line added when the Discord reply already links each card.
_LIST_CITATION_RESERVE = 160
_HASH = re.compile(r"^[0-9a-f]{64}$")
_PROGRESS = frozenset({"doing", "in progress", "active in progress"})
_OPEN_PHRASE = re.compile(
    r"\b(?:still\s+open|not\s+done|(?:tickets?|cards?)\s+(?:are|remain)\s+open)\b",
    re.IGNORECASE,
)
_BOARD_LINK = re.compile(
    r"https://trello\.com/b/(?P<id>[A-Za-z0-9]+)/(?P<slug>[A-Za-z0-9-]+)/?",
    re.IGNORECASE,
)
_INVITE_LINK = re.compile(
    r"https://trello\.com/invite/b/(?P<id>[A-Za-z0-9]+)/(?P<token>[A-Za-z0-9]+)/(?P<slug>[A-Za-z0-9-]+)/?",
    re.IGNORECASE,
)
_BARE_STATE = re.compile(
    r"""(?ix)
    ^\s*
    in\s+
    (?:the\s+)?
    (?P<name>["“'‘]?[A-Za-z0-9][A-Za-z0-9 _-]{0,40}?["”'’]?)
    \s+
    (?:column|list|lane|state|status|stage)
    \s*[?.!]*\s*$
    """
)
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
        if _OPEN_PHRASE.search(text):
            return OPEN_LISTS
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
    if requested == OPEN_LISTS:
        matched = [card for card in usable if _normalize(card.list_name) != "done"]
        if not matched:
            names = ", ".join(_indexed_list_names(usable))
            return AnswerResponse(
                status="no_relevant_information",
                answer=(
                    "No indexed Trello cards are still open. "
                    f"Lists with indexed cards: {names}."
                ),
                confidence=0.0,
                citations=[],
            )
    else:
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
    prefix = _open_summary(matched) if requested == OPEN_LISTS else ""
    shown, hidden = _cards_that_fit(matched, prefix=prefix)
    lines = _answer_lines(shown, prefix=prefix, hidden=hidden)
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


def _list_label(card: TrelloCardRow) -> str:
    return " ".join(redact_secrets(card.list_name).split()) or "Untitled list"


def _open_summary(cards: Sequence[TrelloCardRow]) -> str:
    """Per-list counts for every open card, including ones past the Discord cap."""
    counts: dict[str, int] = {}
    for card in cards:
        label = _list_label(card)
        counts[label] = counts.get(label, 0) + 1
    return ", ".join(f"{name} {count}" for name, count in counts.items())


def _answer_lines(cards: Sequence[TrelloCardRow], *, prefix: str, hidden: int) -> list[str]:
    lines: list[str] = []
    if prefix:
        lines.append(prefix)
        lines.append("")
    lines.extend(_render_groups(cards))
    if hidden:
        lines.append(f"…and {hidden} more.")
    return lines


def _cards_that_fit(
    matched: Sequence[TrelloCardRow], *, prefix: str = ""
) -> tuple[list[TrelloCardRow], int]:
    """Keep every card that fits in a Discord message, then say how many remain."""
    cap = min(len(matched), MAX_LISTED_CARDS)
    chosen = 1 if matched else 0
    for size in range(cap, 0, -1):
        hidden = len(matched) - size
        body = "\n".join(_answer_lines(matched[:size], prefix=prefix, hidden=hidden))
        if len(body) + _LIST_CITATION_RESERVE <= _DISCORD_MESSAGE_LIMIT:
            chosen = size
            break
    shown = list(matched[:chosen])
    return shown, len(matched) - len(shown)


def asks_for_trello_board_link(question: str) -> bool:
    """Match a request for the board URL, not a how-to or a list catalog."""
    text = question.casefold()
    if re.search(r"\bhow\s+(?:do|can|should|to)\b", text):
        return False
    if re.search(r"\btrello\b", text) is None or re.search(r"\bboards?\b", text) is None:
        return False
    return re.search(r"\b(?:link|url|uri|address)\b", text) is not None


def _url_in_text(text: str) -> str | None:
    """Canonical ``/b/<id>/<slug>`` URL. An invite is rewritten and its token dropped."""
    direct = _BOARD_LINK.search(text)
    if direct is not None and "/invite/" not in direct.group(0).casefold():
        return f"https://trello.com/b/{direct.group('id')}/{direct.group('slug')}"
    invite = _INVITE_LINK.search(text)
    if invite is None:
        return None
    return f"https://trello.com/b/{invite.group('id')}/{invite.group('slug')}"


def board_url_from_cards(rows: Sequence[TrelloCardRow]) -> str | None:
    """Canonical ``/b/<id>/<slug>`` URL from card rows, never an invite token."""
    text = "\n".join(
        "\n".join((card.url, card.chunk_text, card.source_file, card.board)) for card in rows
    )
    return _url_in_text(text)


def bare_list_name(question: str) -> str | None:
    """List name from a follow-up that is only ``in <list> state?``."""
    match = _BARE_STATE.match(" ".join(question.split()))
    if match is None:
        return None
    normalized = _normalize(match.group("name").strip("\"'“”‘’"))
    if not normalized or len(normalized) > 40:
        return None
    if normalized in _PROGRESS or _is_progress(normalized):
        return "doing"
    return normalized


def indexed_list_request(name: str, rows: Sequence[TrelloCardRow]) -> str | None:
    """Return ``name`` when it equals an indexed list, including the Doing alias."""
    usable = _usable_cards(rows)
    if any(_list_matches(card.list_name, name) for card in usable):
        return name
    return None


def _board_quote(card: TrelloCardRow, url: str) -> str | None:
    """Cite a line that contains the board URL, never an invite token."""
    for line in card.chunk_text.splitlines():
        stripped = line.strip()
        if not stripped or "ATTI" in stripped or "/invite/" in stripped.casefold():
            continue
        if url in stripped:
            redacted = redact_secrets(stripped).strip()
            if url in redacted and "ATTI" not in redacted:
                return redacted
    if _INVITE_LINK.search(card.chunk_text) or _INVITE_LINK.search(card.url) or url in card.chunk_text:
        quote = _quote(card)
        if quote and "ATTI" not in quote and "/invite/" not in quote.casefold():
            return quote
    return None


def board_link_from_chunk_rows(rows: Sequence[tuple[int, str, str, str]]) -> AnswerResponse | None:
    """Cite a ``trello.com/b/`` line from any chunk. Invite tokens stay out of the answer."""
    ranked: list[tuple[bool, int, str, str, str, str]] = []
    for chunk_id, source_file, digest, chunk_text in rows:
        hashed = (digest or "").strip().lower()
        if chunk_id < 1 or not source_file or _HASH.fullmatch(hashed) is None:
            continue
        url = _url_in_text(chunk_text)
        if url is None or "ATTI" in url or "/invite/" in url:
            continue
        quote = _line_with_url(chunk_text, url)
        if quote is None:
            continue
        direct = "/invite/" not in chunk_text.casefold() and "ATTI" not in chunk_text
        ranked.append((direct, chunk_id, source_file, hashed, url, quote))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (not item[0], item[1]))
    _direct, chunk_id, source_file, digest, url, quote = ranked[0]
    return AnswerResponse(
        status="answered",
        answer=f"The Trello board is {url}.",
        confidence=0.8,
        citations=[
            Citation(
                source_file=source_file,
                chunk_id=chunk_id,
                content_sha256=digest,
                quote=quote,
            )
        ],
    )


def _line_with_url(text: str, url: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or "ATTI" in stripped or "/invite/" in stripped.casefold():
            continue
        if url in stripped:
            redacted = redact_secrets(stripped).strip()
            if url in redacted and "ATTI" not in redacted:
                return redacted
    return None


def trello_board_link_answer(rows: Sequence[TrelloCardRow]) -> AnswerResponse:
    """Return the board URL stored on indexed cards, without an invite token."""
    usable = _usable_cards(rows)
    url = board_url_from_cards(usable)
    if url is None or "ATTI" in url or "/invite/" in url:
        return AnswerResponse(
            status="no_relevant_information",
            answer="The indexed Trello cards do not include a board link.",
            confidence=0.0,
            citations=[],
        )
    for card in usable:
        quote = _board_quote(card, url)
        if quote is None:
            continue
        return AnswerResponse(
            status="answered",
            answer=f"The Trello board is {url}.",
            confidence=0.8,
            citations=[
                Citation(
                    source_file=card.source_file,
                    chunk_id=card.chunk_id,
                    content_sha256=card.content_sha256,
                    quote=quote,
                )
            ],
        )
    return AnswerResponse(
        status="no_relevant_information",
        answer="The indexed Trello cards do not include a board link.",
        confidence=0.0,
        citations=[],
    )


_CARD_HEADER = re.compile(r"^(?:Card|Board|List|URL|Labels|Due):")
_PROJECT_STATUS = re.compile(r"\bstatus\b", re.IGNORECASE)
_ACTIVE_PROJECTS = re.compile(
    r"\b(?:working|work(?:ing)?\s+on|current)\b",
    re.IGNORECASE,
)
_PROJECT_WORD = re.compile(r"\bprojects?\b", re.IGNORECASE)
_STATUS_SKIP = frozenset(
    {
        "project",
        "projects",
        "status",
        "current",
        "currently",
        "bmtech",
        "what",
        "which",
        "the",
        "and",
        "for",
        "our",
        "are",
        "does",
        "how",
    }
)


def asks_for_project_status(question: str) -> bool:
    """A status question about one named project, not a list catalog."""
    return _PROJECT_STATUS.search(question) is not None and _PROJECT_WORD.search(question) is not None


def asks_for_active_projects(question: str) -> bool:
    """Which projects are in progress. A single project's status stays on that path."""
    if asks_for_project_status(question):
        return False
    return _PROJECT_WORD.search(question) is not None and _ACTIVE_PROJECTS.search(question) is not None


def _project_terms(question: str) -> list[str]:
    terms: list[str] = []
    for token in re.findall(r"[a-z0-9]+", question.casefold()):
        if len(token) < 3 or token in _STATUS_SKIP or token in terms:
            continue
        terms.append(token)
    return terms


def _title_has_term(title: str, term: str) -> bool:
    folded = title.casefold()
    if term in folded:
        return True
    stem = term[:-1] if term.endswith("s") and len(term) > 3 else term
    return stem in folded


def project_status_answer(question: str, rows: Sequence[TrelloCardRow]) -> AnswerResponse | None:
    """Card title, list, and link for a project named in the question."""
    terms = _project_terms(question)
    if not terms:
        return None
    matched = [
        card
        for card in _usable_cards(rows)
        if all(_title_has_term(card.title, term) for term in terms)
    ]
    if not matched:
        return None
    lines = [f"Indexed cards for {' '.join(terms)}:"]
    for card in matched:
        title = " ".join(redact_secrets(card.title).split()) or "Untitled card"
        where = _list_label(card)
        if card.url.startswith("https://"):
            lines.append(f"- [{title}]({card.url}) — {where}")
        else:
            lines.append(f"- {title} — {where}")
    return AnswerResponse(
        status="answered",
        answer="\n".join(lines),
        confidence=0.8,
        citations=[_citation(card) for card in matched],
    )


def _card_description(card: TrelloCardRow) -> str:
    lines: list[str] = []
    for raw in card.chunk_text.splitlines():
        line = _strip_marker(raw)
        if not line or _CARD_HEADER.match(line):
            continue
        if line.casefold() == card.title.casefold():
            continue
        lines.append(line)
    return " ".join(lines)


def _strip_marker(text: str) -> str:
    return re.sub(r"^(?:[-*]\s+)+", "", text.strip()).strip()


def _project_chunk_line(chunk: SearchChunk) -> tuple[str, str] | None:
    """A cited project note and a verbatim quote. Bare card headers are skipped."""
    if str((chunk.metadata or {}).get("source_type") or "") == "trello":
        return None
    if not chunk.content_sha256:
        return None
    title = str((chunk.metadata or {}).get("title") or "").strip()
    sentences: list[str] = []
    for raw in chunk.chunk_text.splitlines():
        line = _strip_marker(raw)
        if not line or _CARD_HEADER.match(line) or line not in chunk.chunk_text:
            continue
        sentences.append(line)
    if not sentences:
        return None
    body = " ".join(sentences).strip()
    if body.lower().startswith("card:"):
        return None
    if "project" not in f"{title} {body}".casefold():
        return None
    shown = f"{title}: {body}" if title and body else body
    return shown, sentences[0]


def _is_project_card(card: TrelloCardRow) -> bool:
    return "project" in f"{card.title} {_card_description(card)}".casefold()


def _linked_card(card: TrelloCardRow, label: str) -> str:
    title = " ".join(redact_secrets(card.title).split()) or "Untitled card"
    item = f"[{title}]({card.url})" if card.url.startswith("https://") else title
    description = _card_description(card)
    if description:
        return f"- {item} — {label}: {description}"
    return f"- {item} — {label}"


def active_projects_answer(
    rows: Sequence[TrelloCardRow], chunks: Sequence[SearchChunk]
) -> AnswerResponse | None:
    """Doing cards are active work. Project cards in Backlog are planned, not active."""
    usable = _usable_cards(rows)
    doing = [card for card in usable if _list_matches(card.list_name, "doing")]
    planned = [
        card
        for card in usable
        if _list_matches(card.list_name, "backlog") and _is_project_card(card)
    ]
    lines: list[str] = []
    citations: list[Citation] = []
    if doing:
        lines.append("Active work:")
        for card in doing:
            lines.append(_linked_card(card, "Doing (active)"))
            citations.append(_citation(card))
    if planned:
        lines.append("Backlog (planned), not active:")
        for card in planned:
            lines.append(_linked_card(card, "Backlog (planned)"))
            citations.append(_citation(card))
    for chunk in chunks:
        rendered = _project_chunk_line(chunk)
        if rendered is None or not chunk.content_sha256:
            continue
        shown, quote = rendered
        lines.append(f"- {shown}")
        citations.append(
            Citation(
                source_file=chunk.source_file,
                chunk_id=chunk.id,
                content_sha256=chunk.content_sha256,
                quote=quote,
            )
        )
    if not doing and not planned:
        return None
    if not citations:
        return None
    answer = "\n".join(lines)
    if re.search(r"(?m)^- Card:", answer):
        return None
    return AnswerResponse(status="answered", answer=answer, confidence=0.8, citations=citations)


def answer_is_bare_cards(answer: str) -> bool:
    """True when every line is only a Trello header such as ``Card:``."""
    lines = [re.sub(r"^[-*]\s+", "", line.strip()) for line in answer.splitlines() if line.strip()]
    if not lines:
        return False
    return all(_CARD_HEADER.match(line) for line in lines)
