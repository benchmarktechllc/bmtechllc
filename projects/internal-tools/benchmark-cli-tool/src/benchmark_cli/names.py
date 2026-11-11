"""Roster aliases and typo-tolerant name resolution for Ask and Trello sync.

A question with no close match is returned unchanged. Common English words are
not fuzzy-matched.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_MIN_FUZZY_LENGTH = 4
_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "all",
        "also",
        "and",
        "any",
        "are",
        "assigned",
        "assignee",
        "backlog",
        "been",
        "board",
        "but",
        "can",
        "card",
        "cards",
        "did",
        "doing",
        "done",
        "for",
        "from",
        "have",
        "how",
        "into",
        "list",
        "many",
        "more",
        "not",
        "now",
        "our",
        "please",
        "project",
        "projects",
        "show",
        "status",
        "that",
        "the",
        "their",
        "there",
        "this",
        "ticket",
        "tickets",
        "was",
        "what",
        "whats",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "work",
        "working",
        "you",
        "your",
    }
)


@dataclass(frozen=True, slots=True)
class Person:
    """One roster entry. ``canonical`` is what Ask and the Members line use."""

    canonical: str
    aliases: tuple[str, ...]
    trello_names: tuple[str, ...]

    def labels(self) -> tuple[str, ...]:
        seen: list[str] = []
        for label in (self.canonical, *self.aliases, *self.trello_names):
            if label and label.casefold() not in {item.casefold() for item in seen}:
                seen.append(label)
        return tuple(seen)


@dataclass(frozen=True, slots=True)
class NameHit:
    """One token or bigram resolved to a canonical person or term."""

    surface: str
    canonical: str
    kind: str


@dataclass(frozen=True, slots=True)
class NameResolution:
    """Retrieval expansion for a question. Empty when nothing fuzzy matched."""

    query: str
    note: str
    people: tuple[Person, ...]
    hits: tuple[NameHit, ...]

    @property
    def changed(self) -> bool:
        return bool(self.hits)


def _roster_path() -> Path:
    return Path(__file__).with_name("roster.json")


@lru_cache(maxsize=1)
def load_roster() -> tuple[tuple[Person, ...], tuple[str, ...]]:
    """People and distinctive terms shipped with the package."""
    payload = json.loads(_roster_path().read_text(encoding="utf-8"))
    people = tuple(
        Person(
            canonical=str(item["canonical"]),
            aliases=tuple(str(alias) for alias in item.get("aliases", [])),
            trello_names=tuple(str(name) for name in item.get("trello", [])),
        )
        for item in payload.get("people", [])
    )
    terms = tuple(str(term) for term in payload.get("terms", []))
    return people, terms


def person_for_trello_name(trello_name: str) -> Person | None:
    """Roster entry whose Trello full name or username is this member."""
    folded = trello_name.casefold()
    for person in load_roster()[0]:
        if any(name.casefold() == folded for name in person.trello_names):
            return person
        if any(name.casefold() == folded for name in person.labels()):
            return person
    return None


def render_member(trello_name: str) -> str:
    """``Dante Paniccia (aka Dante, Dantino12)``. Unknown members stay as stored."""
    person = person_for_trello_name(trello_name)
    if person is None:
        return trello_name
    akas = [label for label in person.labels() if label.casefold() != person.canonical.casefold()]
    if not akas:
        return person.canonical
    return f"{person.canonical} (aka {', '.join(akas)})"


def render_members(names: tuple[str, ...] | list[str]) -> str:
    return ", ".join(render_member(name) for name in names)


def split_members_field(body: str) -> list[str]:
    """Split a Members line. Commas inside ``(aka ...)`` stay in one member."""
    parts: list[str] = []
    buffer: list[str] = []
    depth = 0
    for char in body:
        if char == "(":
            depth += 1
        elif char == ")" and depth:
            depth -= 1
        if char == "," and depth == 0:
            piece = "".join(buffer).strip()
            if piece:
                parts.append(piece)
            buffer = []
            continue
        buffer.append(char)
    tail = "".join(buffer).strip()
    if tail:
        parts.append(tail)
    return parts


def member_matches_person(member_text: str, person: Person) -> bool:
    """True when this member field is that person, including an aka list."""
    for label in person.labels():
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(label)}(?![A-Za-z0-9])", member_text, re.IGNORECASE):
            return True
    return False


def _damerau(left: str, right: str) -> int:
    """Damerau-Levenshtein distance. One adjacent swap counts as one edit."""
    if left == right:
        return 0
    rows = len(left) + 1
    cols = len(right) + 1
    score = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        score[i][0] = i
    for j in range(cols):
        score[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if left[i - 1] == right[j - 1] else 1
            score[i][j] = min(
                score[i - 1][j] + 1,
                score[i][j - 1] + 1,
                score[i - 1][j - 1] + cost,
            )
            if i > 1 and j > 1 and left[i - 1] == right[j - 2] and left[i - 2] == right[j - 1]:
                score[i][j] = min(score[i][j], score[i - 2][j - 2] + 1)
    return score[-1][-1]


def _candidates(extra_terms: tuple[str, ...] | list[str]) -> list[tuple[str, str, Person | None]]:
    """(folded label, canonical, person or None for a plain term)."""
    people, terms = load_roster()
    found: list[tuple[str, str, Person | None]] = []
    seen: set[str] = set()
    for person in people:
        for label in person.labels():
            key = label.casefold()
            if key in seen:
                continue
            seen.add(key)
            found.append((key, person.canonical, person))
    for term in (*terms, *extra_terms):
        key = term.casefold()
        if len(key) < _MIN_FUZZY_LENGTH or key in seen or key in _STOPWORDS:
            continue
        seen.add(key)
        found.append((key, term, None))
    return found


def _acceptable(surface: str, label: str) -> bool:
    if surface == label:
        return True
    if surface in _STOPWORDS or len(surface) < _MIN_FUZZY_LENGTH or len(label) < _MIN_FUZZY_LENGTH:
        return False
    distance = _damerau(surface, label)
    if distance == 0:
        return True
    if distance == 1 and min(len(surface), len(label)) >= _MIN_FUZZY_LENGTH:
        return True
    return False


def resolve_question(question: str, extra_terms: tuple[str, ...] | list[str] = ()) -> NameResolution:
    """Expand a question with canonical names. No hits leaves the question as-is."""
    text = " ".join(question.split())
    tokens = _TOKEN.findall(text)
    if not tokens:
        return NameResolution(text, "", (), ())
    folded = [token.casefold() for token in tokens]
    spans: list[tuple[int, int, str]] = [(index, index + 1, folded[index]) for index in range(len(folded))]
    spans.extend(
        (index, index + 2, f"{folded[index]} {folded[index + 1]}") for index in range(len(folded) - 1)
    )
    catalog = _candidates(extra_terms)
    hits: list[NameHit] = []
    used: set[int] = set()
    people: list[Person] = []
    # Longer spans first so "Fanuel Mengistu" wins over "Fanuel".
    for start, end, surface in sorted(spans, key=lambda item: item[1] - item[0], reverse=True):
        if any(index in used for index in range(start, end)):
            continue
        best: tuple[str, str, Person | None] | None = None
        for label, canonical, person in catalog:
            if not _acceptable(surface, label):
                continue
            if best is None or _damerau(surface, label) < _damerau(surface, best[0]):
                best = (label, canonical, person)
        if best is None:
            continue
        _label, canonical, person = best
        used.update(range(start, end))
        if person is not None and person not in people:
            people.append(person)
        already = surface == canonical.casefold() or (
            person is not None and surface == person.canonical.casefold()
        )
        if already:
            continue
        original = " ".join(tokens[start:end])
        hits.append(NameHit(original, canonical, "person" if person is not None else "term"))
    if not hits:
        return NameResolution(text, "", (), ())
    suffix = []
    for hit in hits:
        if hit.canonical.casefold() not in text.casefold() and hit.canonical not in suffix:
            suffix.append(hit.canonical)
    query = text if not suffix else f"{text} {' '.join(suffix)}"
    note = " ".join(f"'{hit.surface}' interpreted as {hit.canonical}." for hit in hits)
    return NameResolution(query, note, tuple(people), tuple(hits))
