"""Answer Trello list-membership questions from indexed cards.

Hybrid retrieval returns only a few chunks, so it cannot enumerate a board
list. A narrow membership question is answered from the first indexed chunk
of each card. Setup, history, and other Trello questions stay on the normal
grounded path.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .ingestion.clean import redact_secrets
from .models import AnswerResponse, Citation
from .roster import Person, load_people, person_by_label, person_in_text, roster_payload
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
    r"\b(?:why|set[\s-]?up|setup|configure|config|integrat\w*|connector|"
    r"api|sync\w*|ingest\w*|tokens?|webhooks?|work(?:ing|s)?)\b",
    re.IGNORECASE,
)
_PROCEDURE = re.compile(r"\bhow\s+(?:do|does|can|should|to|did)\b", re.IGNORECASE)
_HISTORY = re.compile(r"\b(?:add|added|did)\b", re.IGNORECASE)
_FILLER = re.compile(
    r"\b(?:right now|currently|today|please|just|still|on trello|on the board)\b",
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
    if not text or _PROCEDURE.search(text) or _NOT_MEMBERSHIP.search(text):
        return None
    folded = text.casefold()
    if re.search(r"\b(?:cards?|tickets?|trello)\b", folded) is None:
        return None
    if re.search(r"\b(?:which|what|list|show|how many)\b", folded) is None:
        return None
    if _OPEN_PHRASE.search(text):
        return OPEN_LISTS
    match = _LIST_TAIL.search(text)
    if match is None:
        return None
    normalized = _clean_list_name(match.group("name"))
    if not normalized or len(normalized) > 40:
        return None
    if normalized in _PROGRESS or _is_progress(normalized) or (
        normalized == "progress" and re.search(r"\bin\s+progress\b", folded)
    ):
        return "doing"
    if normalized in {"done", "completed", "closed", "finished"}:
        return "done"
    return normalized


def trello_list_answer(
    requested: str,
    rows: Sequence[TrelloCardRow],
    *,
    fit_to_discord: bool = True,
) -> AnswerResponse:
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
    shown, hidden = _cards_that_fit(matched, prefix=prefix, fit_to_discord=fit_to_discord)
    lines = _answer_lines(shown, prefix=prefix, hidden=hidden)
    return AnswerResponse(
        status="answered",
        answer="\n".join(lines),
        confidence=0.8,
        citations=[_citation(card) for card in shown],
    )


def _clean_list_name(name: str) -> str:
    """Drop filler and a trailing list/column word from a captured list name."""
    text = _FILLER.sub(" ", name)
    text = re.sub(
        r"\b(?:column|list|lane|state|status|stage|tickets?|cards?)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    return _normalize(text)


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
    matched: Sequence[TrelloCardRow],
    *,
    prefix: str = "",
    fit_to_discord: bool = True,
) -> tuple[list[TrelloCardRow], int]:
    """Keep every card that fits in one Discord message, unless the caller will split."""
    if not fit_to_discord:
        return list(matched), 0
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


_BOARD_SIGNAL = re.compile(
    r"\b(?:trello|tickets?|cards?|backlog|doing|done|sev\s*\d|in progress|in-progress|"
    r"completed|closed|finished|assigned|assignee|due|urgent)\b|\bwhat(?:'s| is)\s+in\b",
    re.IGNORECASE,
)
_WHO_OWNS = re.compile(
    r"\bwho(?:'s| is| are)?\s+(?:working on|assigned to)\b",
    re.IGNORECASE,
)
_BOARD_OVERVIEW = re.compile(
    r"\bwhat(?:'s| is)\s+on\s+(?:the\s+)?(?:trello\s+)?board\b|\bsummarize the board\b",
    re.IGNORECASE,
)
_URGENT = re.compile(r"\b(?:most\s+urgent|urgent\s+tickets?|highest\s+severity)\b", re.IGNORECASE)
_KB_FAMILY = re.compile(
    r"\b(?:knowledge\s+base|kb|docs|doc\s+site|documentation|portal)\b",
    re.IGNORECASE,
)
_SNAPSHOT_SYSTEM = (
    "Answer only from the JSON board snapshot. "
    "Every card title and URL you write must appear in the snapshot. "
    "Stated list counts must match the snapshot. "
    "Sev 1 is more urgent than Sev 2, and Sev 2 is more urgent than Sev 3. "
    "When asked what is on the board, give every list count and the open cards. "
    "When members are absent, say assignees are not indexed yet and name the card's list. "
    "The thread opener is context, not the question to answer. Do not invent cards."
)
_DEFINITION = re.compile(r"\bwhat(?:'s| is)\s+trello\b", re.IGNORECASE)
_DONE_WORD = re.compile(r"\b(?:completed|closed|finished)\b|\bare\s+done\b|\bbeen\s+completed\b", re.IGNORECASE)
_DOING_WORD = re.compile(
    r"\b(?:in progress|in-progress|active in progress)\b",
    re.IGNORECASE,
)


_CURRENT_WORK = re.compile(
    r"\b(?:currently working|working on|everyone working)\b|\bworking\b",
    re.IGNORECASE,
)


def asks_for_current_work(question: str) -> bool:
    """Tickets being worked on now. A named card's assignee stays on that path."""
    if _WHO_OWNS.search(question) or _PROCEDURE.search(question) or _HISTORY.search(question):
        return False
    if _CURRENT_WORK.search(question) is None:
        return False
    return re.search(r"\b(?:tickets?|cards?|trello|everyone)\b", question, re.IGNORECASE) is not None


def board_question(question: str, thread_question: str | None = None) -> bool:
    """Whether this question, or a Trello thread follow-up, should use the card index."""
    if _BOARD_OVERVIEW.search(question):
        return True
    if asks_for_current_work(question) or re.search(r"\bblocked\b", question, re.IGNORECASE):
        return True
    if _WHO_OWNS.search(question) and _PROCEDURE.search(question) is None and _HISTORY.search(question) is None:
        return True
    if _blocked_board_question(question):
        return False
    if (
        requested_trello_list(question) is not None
        or bare_list_name(question) is not None
        or asks_for_trello_board_link(question)
    ):
        return True
    if _BOARD_SIGNAL.search(question):
        return True
    opener = thread_question or ""
    if opener and not _blocked_board_question(opener) and _BOARD_SIGNAL.search(opener) and _BOARD_SIGNAL.search(question):
        return True
    return False


def _blocked_board_question(question: str) -> bool:
    if _PROCEDURE.search(question) or _HISTORY.search(question) or _DEFINITION.search(question):
        return True
    return _NOT_MEMBERSHIP.search(question) is not None


def _members_line(card: TrelloCardRow) -> str:
    for line in card.chunk_text.splitlines():
        if line.strip().lower().startswith("members:"):
            return line.split(":", 1)[1].strip()
    return ""


def _members_of(card: TrelloCardRow) -> list[str]:
    line = _members_line(card)
    if not line:
        return []
    parts = [part.strip() for part in line.replace(";", ",").split(",") if part.strip()]
    return parts


def member_records(card: TrelloCardRow) -> list[dict[str, object]]:
    """Roster names on this card. Unknown member text stays as written."""
    line = _members_line(card)
    if not line:
        return []
    found = [person for person in roster_people_on_line(line)]
    records = [
        {"name": person.canonical, "aliases": list(person.aliases), "trello": list(person.trello)}
        for person in found
    ]
    if records:
        return records
    return [{"name": part, "aliases": [], "trello": [part]} for part in _members_of(card)]


def roster_people_on_line(line: str) -> list[Person]:
    """Roster people whose names appear on one Members line."""
    return [person for person in load_people() if person_in_text(person, line)]


def cards_for_person(rows: Sequence[TrelloCardRow], person_name: str | None) -> list[TrelloCardRow]:
    """That person's cards, Done included, active lists before Done."""
    person = person_by_label(person_name)
    if person is None:
        return []
    matched = [
        card
        for card in _usable_cards(rows)
        if person_in_text(person, _members_line(card))
    ]
    return sorted(matched, key=lambda card: _normalize(card.list_name) == "done")


def person_card_payload(rows: Sequence[TrelloCardRow], person_name: str | None) -> list[dict[str, object]]:
    """Compact cards for the person filter. Active lists come first."""
    return [_card_brief(card) for card in cards_for_person(rows, person_name)]


def _current_only(question: str) -> bool:
    """A question about what is happening now, not the full ticket history."""
    return re.search(
        r"\b(?:currently|current|right now|at the moment)\b",
        question,
        re.IGNORECASE,
    ) is not None


def _scoped_person_cards(
    rows: Sequence[TrelloCardRow], person_name: str | None, question: str
) -> list[TrelloCardRow]:
    cards = cards_for_person(rows, person_name)
    if not _current_only(question):
        return cards
    active = [card for card in cards if _normalize(card.list_name) != "done"]
    return active or cards


def person_cards_answer(
    rows: Sequence[TrelloCardRow], person_name: str | None, question: str = ""
) -> AnswerResponse | None:
    """Every indexed card for one resolved person. Active lists come first.

    A question about what is happening now stays on the active lists. Any other
    person question includes Done.
    """
    person = person_by_label(person_name)
    cards = _scoped_person_cards(rows, person_name, question)
    if person is None or not cards:
        return None
    lines = [f"{person.canonical}'s tickets:", ""]
    lines.extend(f"{_bullet(card)} — {_list_label(card)}" for card in cards)
    return AnswerResponse(
        status="answered",
        answer="\n".join(lines),
        confidence=0.8,
        citations=[_citation(card) for card in cards],
    )


def titles_outside_person(
    answer: str, rows: Sequence[TrelloCardRow], person_name: str | None
) -> list[str]:
    """Board titles named in the answer that are not this person's cards."""
    if person_by_label(person_name) is None:
        return []
    allowed = {card.title.casefold() for card in cards_for_person(rows, person_name)}
    folded = answer.casefold()
    foreign: list[str] = []
    for card in _usable_cards(rows):
        title = card.title.casefold()
        if title and title in folded and title not in allowed and card.title not in foreign:
            foreign.append(card.title)
    return foreign


def _card_brief(card: TrelloCardRow) -> dict[str, object]:
    return {
        "title": card.title,
        "list": card.list_name,
        "url": card.url,
        "chunk_id": card.chunk_id,
        "source_file": card.source_file,
        "content_sha256": card.content_sha256,
        "quote": _quote(card),
        "labels": _labels_of(card),
        "members": member_records(card),
    }


def _has_due(card: TrelloCardRow) -> bool:
    return any(line.strip().lower().startswith("due:") for line in card.chunk_text.splitlines())


def _labels_of(card: TrelloCardRow) -> list[str]:
    for line in card.chunk_text.splitlines():
        if line.strip().lower().startswith("labels:"):
            return [part.strip() for part in line.split(":", 1)[1].split(",") if part.strip()]
    return []


def _public_excerpt(card: TrelloCardRow) -> str:
    """Description text. A line that only repeats the title is omitted."""
    text = _card_description(card).strip()
    if not text or _normalize(text) == _normalize(card.title):
        return ""
    return text


def _sev_number(text: str) -> int | None:
    match = re.search(r"\bsev(?:erity)?\s*([1-3])\b", text, re.IGNORECASE)
    if match is None:
        match = re.search(r"\(sev\s*([1-3])\)", text, re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))


def _card_sev_tag(card: TrelloCardRow) -> int | None:
    return _sev_number(" ".join(_labels_of(card)) + " " + card.title)


def snapshot_payload(rows: Sequence[TrelloCardRow]) -> dict[str, object]:
    """Structured board the model may see. Counts come from every indexed card."""
    cards = []
    counts: dict[str, int] = {}
    for card in _usable_cards(rows):
        label = _list_label(card)
        counts[label] = counts.get(label, 0) + 1
        cards.append(
            {
                "title": card.title,
                "list": card.list_name,
                "url": card.url,
                "chunk_id": card.chunk_id,
                "source_file": card.source_file,
                "content_sha256": card.content_sha256,
                "quote": _quote(card),
                "labels": _labels_of(card),
                "members": member_records(card),
                "due": next(
                    (
                        line.split(":", 1)[1].strip()
                        for line in card.chunk_text.splitlines()
                        if line.lower().startswith("due:")
                    ),
                    None,
                ),
                "excerpt": _public_excerpt(card)[:240],
            }
        )
    return {"counts": counts, "cards": cards, "roster": roster_payload()}


def snapshot_messages(
    question: str,
    rows: Sequence[TrelloCardRow],
    thread_question: str | None = None,
) -> list[dict[str, str]]:
    """Model input: the question plus every indexed card. The opener is context only."""
    body = {
        "question": question.strip(),
        "thread_opener": " ".join((thread_question or "").split()),
        "board": snapshot_payload(rows),
    }
    return [
        {"role": "system", "content": _SNAPSHOT_SYSTEM},
        {"role": "user", "content": json.dumps(body, ensure_ascii=True)},
    ]


def answer_matches_snapshot(
    answer: str, rows: Sequence[TrelloCardRow], *, check_counts: bool = True
) -> bool:
    """Cited titles, URLs, list counts, and none-claims have to match the board."""
    payload = snapshot_payload(rows)
    cards = payload["cards"]
    counts = payload["counts"]
    if not isinstance(cards, list) or not isinstance(counts, dict):
        return False
    titles = {str(card["title"]).casefold() for card in cards if isinstance(card, dict)}
    urls = {str(card["url"]) for card in cards if isinstance(card, dict)}
    for url in re.findall(r"https://trello\.com/c/[A-Za-z0-9]+", answer):
        if url not in urls:
            return False
    for title in re.findall(r"\[([^\]]+)\]\(https://trello\.com/c/", answer):
        if title.casefold() not in titles:
            return False
    if check_counts:
        for label, count in counts.items():
            if not isinstance(label, str) or not isinstance(count, int):
                continue
            escaped = re.escape(label)
            for pattern in (
                rf"\b{escaped}\b[^,\n.]{{0,24}}?\b(\d+)\b",
                rf"\b(\d+)\b[^,\n.]{{0,24}}?\b{escaped}\b",
            ):
                for match in re.finditer(pattern, answer, re.IGNORECASE):
                    if int(match.group(1)) != count:
                        return False
            if count and re.search(
                rf"\bno\b[^\n.]{{0,40}}\b{escaped}\b",
                answer,
                re.IGNORECASE,
            ):
                return False
    if re.search(r"\bnone\b[^\n.]{0,40}\bdue\b", answer, re.IGNORECASE):
        if any(isinstance(card, dict) and card.get("due") for card in cards):
            return False
    return True


def draft_is_grounded(answer: str, rows: Sequence[TrelloCardRow]) -> bool:
    """A draft has to name a real card, a true count, or a true none-claim."""
    if not answer_matches_snapshot(answer, rows):
        return False
    payload = snapshot_payload(rows)
    cards = payload["cards"]
    counts = payload["counts"]
    if not isinstance(cards, list) or not cards:
        return True
    folded = answer.casefold()
    if any(str(card.get("title", "")).casefold() in folded for card in cards if isinstance(card, dict)):
        return True
    if re.search(
        r"not indexed yet|no list named|none of the indexed|do not have a due date|have no due date",
        folded,
    ):
        return True
    if isinstance(counts, dict):
        for label, count in counts.items():
            if not isinstance(label, str) or not isinstance(count, int):
                continue
            escaped = re.escape(label)
            if re.search(rf"\b{escaped}\b[^,\n.]{{0,40}}?\b{count}\b", answer, re.IGNORECASE):
                return True
            if re.search(rf"\b{count}\b[^,\n.]{{0,40}}?\b{escaped}\b", answer, re.IGNORECASE):
                return True
    return False


def citations_for_answer(answer: str, rows: Sequence[TrelloCardRow]) -> list[Citation]:
    """Citations for cards the answer actually names."""
    folded = answer.casefold()
    cited: list[Citation] = []
    seen: set[str] = set()
    for card in _usable_cards(rows):
        named = card.title.casefold() in folded or (card.url and card.url in answer)
        if not named or card.card_id in seen:
            continue
        seen.add(card.card_id)
        cited.append(_citation(card))
    return cited


def grounded_draft(draft: AnswerResponse | None, rows: Sequence[TrelloCardRow]) -> AnswerResponse | None:
    """Accept a model draft only when its cards, counts, and citations match the board."""
    if draft is None or not draft.answer.strip() or not draft_is_grounded(draft.answer, rows):
        return None
    if re.search(
        r"\b(?:none of the indexed|do not have a due date|have no due date)\b",
        draft.answer,
        re.IGNORECASE,
    ):
        return AnswerResponse(
            status="no_relevant_information",
            answer=draft.answer.strip(),
            confidence=0.0,
            citations=[],
        )
    citations = citations_for_answer(draft.answer, rows)
    if not citations:
        return None
    return draft.model_copy(update={"citations": citations})


def prefer_validated_draft(
    draft: AnswerResponse | None,
    rows: Sequence[TrelloCardRow],
    deterministic: AnswerResponse,
) -> AnswerResponse:
    """Keep a model draft only when every card it names is on the board."""
    accepted = grounded_draft(draft, rows)
    if accepted is not None:
        return accepted
    return deterministic


def resolve_board_list(question: str, known_lists: Sequence[str]) -> str | None:
    """List name from anywhere in the question. Filler such as ``right now`` is ignored."""
    named = requested_trello_list(question)
    if named is not None:
        return named
    if bare_list_name(question):
        return bare_list_name(question)
    if _blocked_board_question(question):
        return None
    text = " ".join(question.split())
    folded = _FILLER.sub(" ", text.casefold())
    folded = re.sub(r"\s+", " ", folded)
    if _OPEN_PHRASE.search(text) or re.search(r"\bopen\s+(?:tickets?|cards?)\b", folded):
        return OPEN_LISTS
    if _DONE_WORD.search(folded):
        return "done"
    if _DOING_WORD.search(folded):
        return "doing"
    known = sorted({_normalize(name) for name in known_lists if name.strip()}, key=len, reverse=True)
    for name in known:
        if name and re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", folded):
            if name in _PROGRESS or _is_progress(name):
                return "doing"
            if name == "done":
                return "done"
            return name
    match = re.search(r"\bwhat(?:'s| is)\s+in\s+(?:the\s+)?([a-z0-9]+(?:\s+[a-z0-9]+)?)", folded)
    if match:
        guessed = _clean_list_name(match.group(1))
        if guessed and guessed not in known:
            return guessed
    return None


def _focus_terms(question: str) -> list[str]:
    skip = _STATUS_SKIP | {"tell", "about", "card", "cards", "ticket", "tickets", "trello", "any", "open"}
    terms = []
    for token in re.findall(r"[a-z0-9]+", question.casefold()):
        if len(token) < 3 or token in skip or token in terms:
            continue
        terms.append(token)
    return terms


def _card_focus_answer(question: str, rows: Sequence[TrelloCardRow]) -> AnswerResponse | None:
    if re.search(r"\b(?:about|describe)\b", question, re.IGNORECASE) is None:
        return None
    if re.search(r"\bcards?\b", question, re.IGNORECASE) is None:
        return None
    terms = _focus_terms(question)
    if not terms:
        return None
    usable = _usable_cards(rows)
    matched = [card for card in usable if all(_title_has_term(card.title, term) for term in terms)]
    if len(matched) != 1:
        matched = [card for card in usable if any(_title_has_term(card.title, term) for term in terms)]
    if not matched:
        return None
    card = matched[0]
    return _card_detail(card)


def _card_detail(card: TrelloCardRow) -> AnswerResponse:
    link = f"[{card.title}]({card.url})" if card.url.startswith("https://") else card.title
    excerpt = _public_excerpt(card)
    answer = f"{link} is on {_list_label(card)}."
    if excerpt:
        answer = f"{answer}\n\n{excerpt}"
    return AnswerResponse(
        status="answered",
        answer=answer,
        confidence=0.8,
        citations=[_citation(card)],
    )


def _named_card(name: str, rows: Sequence[TrelloCardRow]) -> TrelloCardRow | None:
    usable = _usable_cards(rows)
    exact = [card for card in usable if _normalize(card.title) == _normalize(name)]
    if exact:
        return exact[0]
    terms = [
        token
        for token in re.findall(r"[a-z0-9]+", name.casefold())
        if len(token) >= 3 and token not in {"the", "card", "cards", "ticket", "tickets"}
    ]
    if not terms:
        return None
    matched = [card for card in usable if all(term in card.title.casefold() for term in terms)]
    if len(matched) == 1:
        return matched[0]
    return None


def _who_answer(question: str, rows: Sequence[TrelloCardRow]) -> AnswerResponse | None:
    """Who is working on or assigned to one named card."""
    match = _WHO_OWNS.search(question)
    if match is None:
        return None
    name = question[match.end() :].strip(" ?.!")
    name = re.sub(r"^(?:the|a|an)\s+", "", name, flags=re.IGNORECASE).strip(" ?.!")
    card = _named_card(name, rows)
    if card is None:
        return None
    link = f"[{card.title}]({card.url})" if card.url.startswith("https://") else card.title
    label = _list_label(card)
    members = [str(record["name"]) for record in member_records(card) if record.get("name")]
    if not members:
        if not any(member_records(item) for item in _usable_cards(rows)):
            text = f"Assignees are not indexed yet. {link} is on {label}."
        else:
            text = f"No member is listed on {link} ({label})."
        return AnswerResponse(
            status="answered",
            answer=text,
            confidence=0.8,
            citations=[_citation(card)],
        )
    people = ", ".join(members)
    return AnswerResponse(
        status="answered",
        answer=f"{people} is assigned to {link} on {label}.",
        confidence=0.8,
        citations=[_citation(card)],
    )


def _due_answer(question: str, rows: Sequence[TrelloCardRow]) -> AnswerResponse | None:
    if re.search(r"\bdue\b", question, re.IGNORECASE) is None:
        return None
    usable = _usable_cards(rows)
    dated = [card for card in usable if _has_due(card)]
    if not dated:
        return AnswerResponse(
            status="no_relevant_information",
            answer="None of the indexed Trello cards have a due date.",
            confidence=0.0,
            citations=[],
        )
    return trello_list_answer(OPEN_LISTS, dated, fit_to_discord=False)


def _topic_answer(question: str, rows: Sequence[TrelloCardRow]) -> AnswerResponse | None:
    match = re.search(r"\babout\s+(?:the\s+)?(.+?)\s*\??\s*$", question, re.IGNORECASE)
    if match is None or re.search(r"\bcards?\b", question, re.IGNORECASE):
        return None
    topic = match.group(1).strip(" ?.")
    terms = [token for token in re.findall(r"[a-z0-9]+", topic.casefold()) if len(token) >= 3]
    if not terms:
        return None
    usable = [
        card
        for card in _usable_cards(rows)
        if _normalize(card.list_name) != "done" and _topic_hit(topic, card)
    ]
    if not usable:
        return AnswerResponse(
            status="no_relevant_information",
            answer="No open Trello card matches that description.",
            confidence=0.0,
            citations=[],
        )
    return trello_list_answer(OPEN_LISTS, usable, fit_to_discord=False)


def _topic_hit(topic: str, card: TrelloCardRow) -> bool:
    blob = f"{card.title} {_card_description(card)}"
    if _KB_FAMILY.search(topic):
        return _KB_FAMILY.search(blob) is not None
    terms = [token for token in re.findall(r"[a-z0-9]+", topic.casefold()) if len(token) >= 3]
    return bool(terms) and all(term in blob.casefold() for term in terms)


def _overview_answer(question: str, rows: Sequence[TrelloCardRow]) -> AnswerResponse | None:
    if _BOARD_OVERVIEW.search(question) is None:
        return None
    usable = _usable_cards(rows)
    if not usable:
        return None
    counts: dict[str, int] = {}
    for card in usable:
        label = _list_label(card)
        counts[label] = counts.get(label, 0) + 1
    summary = ", ".join(f"{name} {count}" for name, count in counts.items())
    open_cards = [card for card in usable if _normalize(card.list_name) != "done"]
    lines = [summary, ""]
    lines.extend(_render_groups(open_cards))
    return AnswerResponse(
        status="answered",
        answer="\n".join(lines),
        confidence=0.8,
        citations=[_citation(card) for card in open_cards],
    )


def _urgent_answer(question: str, rows: Sequence[TrelloCardRow]) -> AnswerResponse | None:
    if _URGENT.search(question) is None:
        return None
    usable = _usable_cards(rows)
    sev1 = [card for card in usable if _sev_number(card.list_name) == 1]
    doing = [
        card
        for card in usable
        if _is_progress(_normalize(card.list_name)) and _card_sev_tag(card) == 1 and card not in sev1
    ]
    if not sev1 and not doing:
        return None
    lines = ["Most urgent tickets are Sev 1."]
    if sev1:
        lines.extend(["", "Trello cards in Sev 1:"])
        lines.extend(_bullet(card) for card in sev1)
    if doing:
        lines.extend(["", "Doing cards tagged (sev 1):"])
        lines.extend(_bullet(card) for card in doing)
    return AnswerResponse(
        status="answered",
        answer="\n".join(lines),
        confidence=0.8,
        citations=[_citation(card) for card in sev1 + doing],
    )


def _current_work_answer(question: str, rows: Sequence[TrelloCardRow]) -> AnswerResponse | None:
    """Doing is the list of tickets currently being worked on."""
    if not asks_for_current_work(question):
        return None
    return trello_list_answer("doing", rows, fit_to_discord=False)


def _blocked_work_answer(question: str, rows: Sequence[TrelloCardRow]) -> AnswerResponse | None:
    if re.search(r"\bblocked\b", question, re.IGNORECASE) is None:
        return None
    usable = [
        card
        for card in _usable_cards(rows)
        if "blocked" in card.title.casefold()
        or any("block" in label.casefold() for label in _labels_of(card))
    ]
    if not usable:
        return AnswerResponse(
            status="no_relevant_information",
            answer="No indexed Trello card is tagged blocked.",
            confidence=0.0,
            citations=[],
        )
    return trello_list_answer(OPEN_LISTS, usable, fit_to_discord=False)


def _state_count(question: str, response: AnswerResponse, rows: Sequence[TrelloCardRow]) -> AnswerResponse:
    """Say the list size in words when the question asks how many."""
    if response.status != "answered" or re.search(r"\bhow many\b", question, re.IGNORECASE) is None:
        return response
    known = _indexed_list_names(rows)
    requested = resolve_board_list(question, known)
    if requested is None:
        return response
    usable = _usable_cards(rows)
    if requested == OPEN_LISTS:
        count = sum(1 for card in usable if _normalize(card.list_name) != "done")
        sentence = f"There are {count} open cards."
    else:
        matched = [card for card in usable if _list_matches(card.list_name, requested)]
        label = next((name for name in known if _list_matches(name, requested)), requested.title())
        noun = "card" if len(matched) == 1 else "cards"
        sentence = f"{label} has {len(matched)} {noun}."
    if sentence.casefold() in response.answer.casefold():
        return response
    return response.model_copy(update={"answer": f"{sentence}\n\n{response.answer}"})


def answer_board_question(
    question: str,
    rows: Sequence[TrelloCardRow],
    *,
    thread_question: str | None = None,
    draft: AnswerResponse | None = None,
) -> AnswerResponse | None:
    """Deterministic board answer. A model draft is kept only when it matches the snapshot."""
    if not board_question(question, thread_question) and requested_trello_list(question) is None and bare_list_name(question) is None:
        return None
    known = _indexed_list_names(rows)
    bare = bare_list_name(question)
    if bare and not any(_list_matches(name, bare) for name in known):
        if requested_trello_list(question) is None:
            return None
    deterministic = (
        _who_answer(question, rows)
        or _card_focus_answer(question, rows)
        or _due_answer(question, rows)
        or _topic_answer(question, rows)
        or _overview_answer(question, rows)
        or _urgent_answer(question, rows)
        or _current_work_answer(question, rows)
        or _blocked_work_answer(question, rows)
    )
    if deterministic is None:
        requested = resolve_board_list(question, known)
        if requested is not None:
            deterministic = trello_list_answer(requested, rows, fit_to_discord=False)
    accepted = grounded_draft(draft, rows)
    if accepted is not None and _draft_matches_request(question, accepted, rows):
        return _state_count(question, accepted, rows)
    if deterministic is None:
        return None
    return _state_count(question, deterministic, rows)


def _draft_matches_request(
    question: str, response: AnswerResponse, rows: Sequence[TrelloCardRow]
) -> bool:
    """A validated draft for a named list has to mention that list or one of its cards."""
    known = _indexed_list_names(rows)
    requested = resolve_board_list(question, known)
    if requested is None or requested == OPEN_LISTS:
        return True
    matched = [card for card in _usable_cards(rows) if _list_matches(card.list_name, requested)]
    folded = response.answer.casefold()
    if any(card.title.casefold() in folded for card in matched):
        return True
    label = next((name for name in known if _list_matches(name, requested)), "")
    return bool(label) and label.casefold() in folded
