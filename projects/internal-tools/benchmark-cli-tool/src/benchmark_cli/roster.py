"""Team roster data for Trello member lines and the evidence pack.

This module does not match question phrasing. A cheap model reads the roster
and decides whether a follow-up names someone.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Person:
    """One teammate as stored in ``roster.json``."""

    canonical: str
    aliases: tuple[str, ...]
    trello: tuple[str, ...]

    def labels(self) -> tuple[str, ...]:
        """Canonical name, nicknames, and Trello display names, without duplicates."""
        seen: list[str] = []
        for label in (self.canonical, *self.aliases, *self.trello):
            if label and label not in seen:
                seen.append(label)
        return tuple(seen)


def _roster_path() -> Path:
    return Path(__file__).with_name("roster.json")


@cache
def load_people() -> tuple[Person, ...]:
    """People from the roster file. An unreadable file yields an empty roster."""
    try:
        payload = json.loads(_roster_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    people: list[Person] = []
    for item in payload.get("people", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        canonical = " ".join(str(item.get("canonical") or "").split())
        if not canonical:
            continue
        aliases = tuple(
            " ".join(str(alias).split())
            for alias in item.get("aliases", [])
            if " ".join(str(alias).split())
        )
        trello = tuple(
            " ".join(str(name).split())
            for name in item.get("trello", [])
            if " ".join(str(name).split())
        )
        people.append(Person(canonical, aliases, trello))
    return tuple(people)


def roster_payload() -> list[dict[str, object]]:
    """Compact roster for the model evidence pack."""
    return [
        {"name": person.canonical, "aliases": list(person.aliases), "trello": list(person.trello)}
        for person in load_people()
    ]


def person_by_label(label: str | None) -> Person | None:
    """Exact roster match for a canonical name, nickname, or Trello username."""
    folded = " ".join((label or "").split()).casefold()
    if not folded:
        return None
    for person in load_people():
        if folded in {item.casefold() for item in person.labels()}:
            return person
    return None


def label_in_text(label: str, text: str) -> bool:
    """True when ``label`` appears as its own token sequence in ``text``."""
    if not label or not text:
        return False
    pattern = rf"(?<![A-Za-z0-9]){re.escape(label)}(?![A-Za-z0-9])"
    return re.search(pattern, text, re.IGNORECASE) is not None


def person_in_text(person: Person, text: str) -> bool:
    """True when any of the person's names appears in ``text``."""
    return any(label_in_text(label, text) for label in person.labels())


def render_member(trello_name: str) -> str:
    """Real name plus aliases. An unknown Trello name is left unchanged."""
    person = person_by_label(trello_name)
    if person is None:
        return trello_name
    extras: list[str] = []
    for label in (*person.aliases, *person.trello):
        if label.casefold() == person.canonical.casefold() or label in extras:
            continue
        extras.append(label)
    if not extras:
        return person.canonical
    return f"{person.canonical} ({', '.join(extras)})"


def render_members(names: tuple[str, ...] | list[str]) -> str:
    """Join members with semicolons so commas inside aliases stay intact."""
    return "; ".join(render_member(name) for name in names if str(name).strip())


_NOT_A_NAME = frozenset(
    """
    the what who how which tickets ticket cards card board trello on for yet
    done doing project projects is are was were and with from about this that
    still open completed finished closed working work our your team
    """.split()
)


def _edit_distance(left: str, right: str) -> int:
    if abs(len(left) - len(right)) > 2:
        return 3
    previous = list(range(len(right) + 1))
    for index, char in enumerate(left, start=1):
        current = [index]
        for other_index, other in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[other_index] + 1,
                    previous[other_index - 1] + (char != other),
                )
            )
        previous = current
    return previous[-1]


def _close_token(token: str, label: str) -> bool:
    left, right = token.casefold(), label.casefold()
    if left == right:
        return True
    if min(len(left), len(right)) < 3:
        return False
    limit = 1 if min(len(left), len(right)) <= 5 else 2
    return _edit_distance(left, right) <= limit


@cache
def load_terms() -> tuple[str, ...]:
    """Source names from the roster file, such as Trello."""
    try:
        payload = json.loads(_roster_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    if not isinstance(payload, dict):
        return ()
    return tuple(
        " ".join(str(term).split())
        for term in payload.get("terms", [])
        if " ".join(str(term).split())
    )


def mentioned_person(question: str, thread_question: str | None, proposed: str | None) -> str | None:
    """Canonical person only when the question or thread actually refers to them."""
    person = person_by_label(proposed)
    if person is None:
        return None
    text = f"{question} {thread_question or ''}"
    if person_in_text(person, text):
        return person.canonical
    for token in re.findall(r"[A-Za-z][A-Za-z']{2,}", text):
        if token.casefold() in _NOT_A_NAME:
            continue
        for label in person.labels():
            if _close_token(token, label) or any(
                _close_token(token, part) for part in label.split() if len(part) > 2
            ):
                return person.canonical
    return None


def normalize_known_terms(text: str) -> str:
    """Replace a one-edit typo of a known source name, such as trelo to Trello."""

    def fix(token: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9']+", token):
            return token
        for term in load_terms():
            if _close_token(token, term):
                return term
        return token

    return "".join(fix(part) for part in re.split(r"(\W+)", text))
