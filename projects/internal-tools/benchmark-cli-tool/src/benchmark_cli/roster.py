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
