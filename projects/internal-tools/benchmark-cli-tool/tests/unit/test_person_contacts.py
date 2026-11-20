"""Email mentions for a referenced person. Synthetic names and addresses only."""

from __future__ import annotations

from typing import Any

import pytest
from test_context_first import _cite, _draft, _install, _kb, _pack

from benchmark_cli import agent, person_contacts
from benchmark_cli.roster import Person
from benchmark_cli.storage.postgres import SearchChunk

PERSON = Person("Sam Example", ("Sammy", "Sam", "Ex"), ("sam-trello",))
TEXT = "line one\nline two\nline three\nReach me at sam.example@example.com please\nline five"


def _row(row_id: int = 5, text: str = TEXT) -> SearchChunk:
    meta = {"url": "https://example.com/m/1", "date": "2026-11-01", "title": "Chat", "source_type": "discord"}
    return SearchChunk(row_id, "chat/x.md", 2, text, meta, 0.0, content_sha256="a" * 64)


@pytest.fixture(autouse=True)
def _roster(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(person_contacts, "person_by_label", lambda label: PERSON if label == "Sam Example" else None)


def _rows(monkeypatch: pytest.MonkeyPatch, rows: list[SearchChunk]) -> list[Any]:
    calls: list[Any] = []

    def fake(parts: Any, limit: int = 6) -> list[SearchChunk]:
        calls.append(list(parts))
        return rows

    monkeypatch.setattr(person_contacts, "person_email_rows", fake)
    return calls


def test_name_parts() -> None:
    assert person_contacts.name_parts("Sam Example") == ["example", "sammy", "trello"]
    assert person_contacts.name_parts("Nobody Known") == ["nobody", "known"]


def test_snippet_is_verbatim_with_two_preceding_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    _rows(monkeypatch, [_row()])
    [chunk] = person_contacts.contact_chunks("Sam Example")
    assert chunk.chunk_text == "line two\nline three\nReach me at sam.example@example.com please"
    assert chunk.chunk_text in TEXT
    assert chunk.metadata["contact_for"] == "Sam Example" and chunk.metadata["derived_from_chunk_id"] == 5
    assert chunk.metadata["url"] == "https://example.com/m/1" and chunk.metadata["date"] == "2026-11-01"
    assert chunk.source_file == "chat/x.md"
    assert chunk.id == 8_000_000_000_000 + 50 and 0 < chunk.id < 9_000_000_000_000


def test_ids_distinct_dedupe_and_unrelated_emails(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "a sam.example@example.com b\nother@example.com\nsam.example@example.com b"
    _rows(monkeypatch, [_row(5, text), _row(6, text)])
    chunks = person_contacts.contact_chunks("Sam Example")
    assert len({c.id for c in chunks}) == len(chunks)
    assert all("other@example.com" not in c.chunk_text.split("\n")[-1] for c in chunks)
    first = [c for c in chunks if c.metadata["derived_from_chunk_id"] == 5]
    assert len(first) == 2  # different snippets, same email


def test_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    _rows(monkeypatch, [_row(i, f"sam{i}.example@example.com") for i in range(1, 20)])
    assert len(person_contacts.contact_chunks("Sam Example")) == 6


def test_db_error_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: Any, **_k: Any) -> list[SearchChunk]:
        raise RuntimeError("db down")

    monkeypatch.setattr(person_contacts, "person_email_rows", boom)
    assert person_contacts.contact_chunks("Sam Example") == []


def _patch_person(monkeypatch: pytest.MonkeyPatch, person: str | None) -> None:
    monkeypatch.setattr(agent, "mentioned_person", lambda *_a, **_k: person)


def test_run_agent_adds_person_contacts(monkeypatch: pytest.MonkeyPatch) -> None:
    [chunk] = (_rows(monkeypatch, [_row()]), person_contacts.contact_chunks("Sam Example"))[1]
    seen = _install(monkeypatch, [_draft("ok", _cite(chunk.id, "Reach me at sam.example@example.com please"))])
    _patch_person(monkeypatch, "Sam Example")
    agent.run_agent("What is Sam's email?")
    pack = _pack(seen["calls"][0])
    assert pack["person_contacts"]["person"] == "Sam Example"
    [entry] = pack["person_contacts"]["entries"]
    assert entry["chunk_id"] == chunk.id and entry["source_label"] == "Contact mention"
    assert "sam.example@example.com" in entry["text"]


def test_contacts_alone_are_enough_to_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    _rows(monkeypatch, [_row()])
    seen = _install(monkeypatch, [_draft("ok")])
    _patch_person(monkeypatch, "Sam Example")
    agent.run_agent("What is Sam's email?")
    assert seen["calls"] and "person_contacts" in _pack(seen["calls"][0])


def test_no_person_no_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _rows(monkeypatch, [_row()])
    seen = _install(monkeypatch, [_draft("ok")], hits=[_kb()])
    _patch_person(monkeypatch, None)
    agent.run_agent("What is the portal?")
    assert calls == []
    assert "person_contacts" not in _pack(seen["calls"][0])
