"""Meetings for Ask: detect meeting questions and resolve which meeting is meant.

Everything here is deterministic and makes no model call. Detection only decides whether
the full meeting is added to the context pack; it never forces an answer.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from .models import AnswerResponse
from .storage.postgres import list_meetings

MEETING_TIMEZONE = "America/New_York"
MEETING_SOURCE_PREFIXES = ("fireflies/", "fathom/")
MEETING_NOT_FOUND = "meeting_not_found"
MAX_LISTED_MEETINGS = 10

# Bare "sync", "call", "fireflies" and "fathom" are left out on purpose: they collide
# with code and product questions such as "where is Discord message sync".
_MEETING_WORDS = re.compile(
    r"\b(?:meetings?|sync\s+calls?|stand-?ups?|transcripts?"
    r"|(?:last|latest|our|team|weekly|today'?s|yesterday'?s)\s+call)\b",
    re.IGNORECASE,
)

TRELLO_EXCLUDED_RETRY = (
    "The conversation asked you not to use Trello tickets. Answer without citing a Trello card; "
    "cite meeting, knowledge-base, chat, or saved-answer chunks instead."
)

_TRELLO_NEGATION = re.compile(
    r"\b(?:don'?t|do not|dont|not|without|ignore|skip|exclude)\b[^.?!\n]{0,30}"
    r"\b(?:tickets?|trello|cards?|board)\b",
    re.IGNORECASE,
)
_ONLY_SOURCE = re.compile(
    r"\bonly\s+(?:on|from|in|using)\s+(?:[\w']+\s+){0,3}(?:meeting|call|transcript|sync|summary)"
    r"|\b(?:just|solely)\s+(?:the\s+)?(?:meeting|transcript)",
    re.IGNORECASE,
)

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}  # fmt: skip
_MONTH = r"(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_ORDINAL = r"(?:st|nd|rd|th)?"
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTH_DAY = re.compile(rf"\b{_MONTH}\.?\s+(?P<day>\d{{1,2}}){_ORDINAL}\b(?:,?\s+(?P<year>\d{{4}}))?", re.IGNORECASE)
_DAY_MONTH = re.compile(rf"\b(?P<day>\d{{1,2}}){_ORDINAL}\s+(?:of\s+)?{_MONTH}\b\.?(?:,?\s+(?P<year>\d{{4}}))?", re.IGNORECASE)
_NUMERIC = re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?:/(\d{2}|\d{4}))?(?![\d/])")
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_LAST_WEEKDAY = re.compile(rf"\blast\s+({'|'.join(_WEEKDAYS)})\b", re.IGNORECASE)
_RECENCY = re.compile(r"\b(?:last|latest|most\s+recent|previous)\b", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+")
_NOT_TITLE_WORDS = frozenset(
    {
        "a", "an", "and", "are", "did", "do", "does", "for", "from", "in", "is", "of", "on", "our",
        "the", "to", "was", "we", "were", "what", "when", "who", "which", "with", "about", "discuss",
        "discussed", "last", "latest", "most", "recent", "previous", "todos", "todo", "yesterday", "today",
    }
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class Meeting:
    source_file: str
    title: str
    date: str
    url: str = ""


@dataclass(frozen=True, slots=True)
class MeetingResolution:
    meeting: Meeting | None = None
    requested: date | None = None

    @property
    def not_found(self) -> bool:
        return self.meeting is None and self.requested is not None


def asks_about_meeting(question: str) -> bool:
    """True when the question is about a meeting, call, stand-up or transcript."""
    return _MEETING_WORDS.search(question or "") is not None


def meeting_context(question: str, thread_question: str | None = None) -> str:
    """The question plus its thread starter, or just the question when there is no distinct parent."""
    current = " ".join((question or "").split())
    parent = " ".join((thread_question or "").split())
    if not parent or parent == current:
        return current
    return f"{current}\n{parent}"


def asks_about_meeting_in_context(question: str, thread_question: str | None = None) -> bool:
    return asks_about_meeting(meeting_context(question, thread_question))


def excludes_trello(question: str, thread_question: str | None = None) -> bool:
    """True when the question (or its thread starter) tells us to leave Trello tickets out."""
    text = meeting_context(question, thread_question)
    return _TRELLO_NEGATION.search(text) is not None or _ONLY_SOURCE.search(text) is not None


def today_in_new_york(now: datetime | None = None) -> date:
    zone = ZoneInfo(MEETING_TIMEZONE)
    return (now.astimezone(zone) if now is not None else datetime.now(zone)).date()


def _make_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _most_recent(month: int, day: int, today: date) -> date | None:
    """The latest month/day that is not after ``today``."""
    for year in (today.year, today.year - 1, today.year - 2, today.year - 3, today.year - 4):
        candidate = _make_date(year, month, day)  # a Feb 29 needs a leap year
        if candidate is not None and candidate <= today:
            return candidate
    return None


def _with_year(month: int, day: int, year: str | None, today: date) -> date | None:
    if year is None:
        return _most_recent(month, day, today)
    value = int(year)
    return _make_date(value + 2000 if value < 100 else value, month, day)


def parse_requested_date(question: str, today: date) -> date | None:
    """The calendar date the question names, resolved against ``today`` (New York)."""
    text = question or ""
    iso = _ISO.search(text)
    if iso is not None:
        return _make_date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
    for pattern in (_MONTH_DAY, _DAY_MONTH):
        found = pattern.search(text)
        if found is not None:
            month = _MONTHS[found.group("month")[:3].lower()]
            return _with_year(month, int(found.group("day")), found.group("year"), today)
    numeric = _NUMERIC.search(text)
    if numeric is not None:
        return _with_year(int(numeric.group(1)), int(numeric.group(2)), numeric.group(3), today)
    lowered = text.lower()
    if re.search(r"\byesterday(?:'?s)?\b", lowered):
        return today - timedelta(days=1)
    if re.search(r"\btoday(?:'?s)?\b", lowered):
        return today
    weekday = _LAST_WEEKDAY.search(text)
    if weekday is not None:
        back = (today.weekday() - _WEEKDAYS.index(weekday.group(1).lower())) % 7
        return today - timedelta(days=back or 7)
    return None


def _meeting_date(meeting: Meeting) -> date | None:
    try:
        return date.fromisoformat(meeting.date)
    except ValueError:
        return None


def _newest(meetings: list[Meeting]) -> Meeting | None:
    dated = [(parsed, item) for item in meetings if (parsed := _meeting_date(item)) is not None]
    if not dated:
        return None
    return max(dated, key=lambda pair: (pair[0], pair[1].source_file))[1]


def _title_words(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.lower()) if word not in _NOT_TITLE_WORDS}


def resolve_meeting(question: str, meetings: list[Meeting], today: date) -> MeetingResolution:
    """Pick the meeting a question means, or report the date that has none.

    A stored date may be the UTC day after the local one, so d and d + 1 both match d.
    """
    requested = parse_requested_date(question, today)
    if requested is not None:
        for wanted in (requested, requested + timedelta(days=1)):
            hits = [item for item in meetings if _meeting_date(item) == wanted]
            if hits:
                return MeetingResolution(_newest(hits), requested)
        return MeetingResolution(None, requested)
    if _RECENCY.search(question or ""):
        words = _title_words(question)
        overlapping = [item for item in meetings if words & _title_words(item.title)]
        return MeetingResolution(_newest(overlapping) or _newest(meetings), None)
    return MeetingResolution()


def resolve_meeting_in_context(
    question: str, thread_question: str | None, meetings: list[Meeting], today: date
) -> MeetingResolution:
    """Resolve from the question first so its own date wins; fall back to the thread starter."""
    resolution = resolve_meeting(question, meetings, today)
    parent = " ".join((thread_question or "").split())
    if resolution.meeting is not None or resolution.requested is not None or not parent:
        return resolution
    return resolve_meeting(parent, meetings, today)


def _user_turns(history: Sequence[Any] | None) -> list[Any]:
    """Earlier user turns, newest first."""
    return [turn for turn in reversed(list(history or ())) if getattr(turn, "role", "") == "user"]


def asks_about_meeting_in_turns(question: str, history: Sequence[Any] | None = None) -> bool:
    """True when the message, or an earlier user turn, is about a meeting."""
    if asks_about_meeting(question):
        return True
    return any(asks_about_meeting(turn.text) for turn in _user_turns(history))


def resolve_meeting_in_turns(
    question: str,
    history: Sequence[Any] | None,
    meetings: list[Meeting],
    today: date,
) -> MeetingResolution:
    """The current message first, then earlier user turns newest first.

    A relative date in an earlier turn ("yesterdays") resolves against the New York date
    that turn was sent, so a follow-up posted the next day keeps the same meeting.
    """
    resolution = resolve_meeting(question, meetings, today)
    if resolution.meeting is not None or resolution.requested is not None:
        return resolution
    for turn in _user_turns(history):
        sent = getattr(turn, "sent_at", None)
        day = today_in_new_york(sent) if sent is not None else today
        resolution = resolve_meeting(turn.text, meetings, day)
        if resolution.meeting is not None or resolution.requested is not None:
            return resolution
    return MeetingResolution()


def indexed_meetings(source_file: str | None = None) -> list[Meeting]:
    return [Meeting(*row) for row in list_meetings(source_file)]


def _long_date(value: date) -> str:
    return f"{value:%B} {value.day}, {value.year}"


def meeting_not_found_response(requested: date, meetings: list[Meeting]) -> AnswerResponse:
    """Deterministic reply for a named date with no indexed meeting. No model, no citations."""
    ordered = sorted(
        (item for item in meetings if _meeting_date(item) is not None),
        key=lambda item: (item.date, item.source_file),
        reverse=True,
    )[:MAX_LISTED_MEETINGS]
    text = f"I have no meeting indexed for {_long_date(requested)}."
    if ordered:
        listing = ", ".join(f"{item.title or 'Untitled meeting'} ({item.date})" for item in ordered)
        text += f" Indexed meetings: {listing}."
    else:
        text += " No meetings are indexed yet."
    return AnswerResponse(status="no_relevant_information", answer=text, confidence=0.0, citations=[])


def _clean(value: Any, limit: int = 120) -> str:
    return " ".join(str(value or "").replace('"', "'").split())[:limit]


def meeting_label(metadata: dict[str, Any]) -> str | None:
    """``Meeting "<title>" on <date>, <section>``, or None when the title or date is missing."""
    title, when = _clean(metadata.get("title")), _clean(metadata.get("date"), 40)
    if not title or not when:
        return None
    section = _clean(metadata.get("section"), 40)
    label = f'Meeting "{title}" on {when}'
    return f"{label}, {section}" if section else label


def meeting_for_source(source_file: str) -> Meeting | None:
    """Indexed meeting for a ``fireflies/`` or ``fathom/`` source file; None otherwise or on error."""
    if not source_file.startswith(MEETING_SOURCE_PREFIXES):
        return None
    try:
        found = indexed_meetings(source_file)
    except Exception:
        return None
    return found[0] if found else None


MEETING_RETRY_PROMPT = (
    "Your prior answer was rejected. Fix only the problems listed, then answer again from the "
    "supplied context. If the context answers the question, answer it. Cite each claim with ONE "
    "single line copied exactly from the context: never join lines, never use ellipses, never "
    "paraphrase a quote."
)

_QUOTE_SPLIT = re.compile(r"\n|\.\.\.|\u2026| - ")
_BOLD_HEADER = re.compile(r"^\*\*[^*]+\*\*$")
_ACTION_ITEMS = re.compile(r"^\W*action items?\W*$", re.IGNORECASE)
MIN_FRAGMENT_CHARACTERS = 12


def _is_meeting_source(source_file: str) -> bool:
    return source_file.startswith(MEETING_SOURCE_PREFIXES)


def _squashed(text: str) -> str:
    return " ".join(text.split()).casefold()


def repair_meeting_citations(response: AnswerResponse, evidence: dict[int, Any]) -> AnswerResponse | None:
    """Split a quote stitched from non-adjacent meeting lines into verbatim single-line citations.

    Only ``fireflies/`` and ``fathom/`` chunks are touched. Every kept quote is an exact substring of
    its chunk. None when a draft had citations and none survive.
    """
    if response.status == "no_relevant_information" or not response.citations:
        return response
    kept = []
    for citation in response.citations:
        chunk = evidence.get(citation.chunk_id)
        if chunk is None or not _is_meeting_source(chunk.source_file) or not chunk.content_sha256:
            kept.append(citation)
            continue
        text = chunk.chunk_text
        if citation.quote in text or _squashed(citation.quote) in _squashed(text):
            kept.append(citation)
            continue
        seen: set[str] = set()
        for raw in _QUOTE_SPLIT.split(citation.quote):
            fragment = raw.strip()
            if fragment in seen or fragment not in text:
                continue
            if _BOLD_HEADER.match(fragment) or len(fragment) >= MIN_FRAGMENT_CHARACTERS:
                seen.add(fragment)
                kept.append(citation.model_copy(update={"quote": fragment}))
    if not kept:
        return None
    return response.model_copy(update={"citations": kept})


def action_items_by_person(text: str) -> dict[str, list[str]]:
    """Person -> item lines from the Summary 'Action items:' block (``**Name**`` then item lines)."""
    owners: dict[str, list[str]] = {}
    inside = False
    person: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not inside:
            inside = bool(_ACTION_ITEMS.match(line.rstrip(":").strip()) or _ACTION_ITEMS.match(line))
            continue
        if not line:
            continue
        if line.startswith("#"):
            break
        if _BOLD_HEADER.match(line):
            person = line.strip("*").strip()
            owners.setdefault(person, [])
        elif person is not None:
            owners[person].append(line)
    return owners


def _names_person(answer: str, person: str) -> bool:
    folded = answer.casefold()
    words = [word for word in re.findall(r"[a-z']+", person.casefold()) if len(word) > 1]
    return person.casefold() in folded or bool(words and words[0] in folded)


def meeting_owner_conflict(response: AnswerResponse, evidence: dict[int, Any]) -> bool:
    """True when a cited action-item line belongs to a person the answer does not name."""
    for citation in response.citations:
        chunk = evidence.get(citation.chunk_id)
        if chunk is None or not _is_meeting_source(chunk.source_file):
            continue
        owners = action_items_by_person(chunk.chunk_text)
        quote = citation.quote.strip()
        if not quote:
            continue
        listed: dict[str, set[str]] = {}
        for person, items in owners.items():
            for item in items:
                if item in quote or (len(quote) >= MIN_FRAGMENT_CHARACTERS and quote in item):
                    listed.setdefault(item, set()).add(person)
        for people in listed.values():
            if not any(_names_person(response.answer, person) for person in people):
                return True
    return False
