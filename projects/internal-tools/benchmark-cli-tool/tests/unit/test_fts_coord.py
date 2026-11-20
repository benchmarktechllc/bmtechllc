"""Coordination-scored full-text search: group parsing, roster expansion, SQL construction."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

from benchmark_cli import roster
from benchmark_cli.storage import postgres


class _Person:
    canonical = "Sam Example"

    def labels(self) -> tuple[str, ...]:
        return ("samexampleson", "sam99")


@pytest.fixture(autouse=True)
def _fake_roster(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(roster, "load_people", lambda: (_Person(),))


class _Conn:
    def __init__(self, tsquery_text: str | None) -> None:
        self.text = tsquery_text
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    def cursor(self) -> _Conn:
        return self

    def __enter__(self) -> _Conn:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, statement: str, parameters: tuple[Any, ...]) -> None:
        self.executed.append((statement, parameters))

    def fetchone(self) -> tuple[str] | None:
        return None if self.text is None else (self.text,)

    def fetchall(self) -> list[tuple[Any, ...]]:
        return []


def test_plain_words_untouched_and_deduped() -> None:
    conn = _Conn("'budget' & 'report' & 'budget'")
    assert postgres._fts_query_groups("x", conn) == ["'budget'", "'report'"]


def test_prefix_expands_to_all_person_words() -> None:
    group = postgres._fts_query_groups("x", _Conn("'sam'"))[0]
    assert group == "('example':* | 'samexampleson':*)"


def test_longer_lexeme_with_person_prefix_expands() -> None:
    assert "'samexampleson':*" in postgres._fts_query_groups("x", _Conn("'samexamplesonx'"))[0]


def test_edit_distance_one_expands() -> None:
    assert "'example':*" in postgres._fts_query_groups("x", _Conn("'exampla'"))[0]
    assert postgres._fts_query_groups("x", _Conn("'zzzzz'")) == ["'zzzzz'"]


def test_unsafe_lexemes_dropped() -> None:
    assert postgres._fts_query_groups("x", _Conn("'a\\b' & 'ok1' & 'x:*'")) == ["'ok1'"]


@pytest.mark.parametrize("text", [None, "", "!'foo'", "'a' | 'b'", "'a' <-> 'b'", "'a' & !'b'", "'a:*'"])
def test_unexpected_or_empty_falls_back(text: str | None) -> None:
    assert postgres._fts_query_groups("x", _Conn(text)) == []


def test_fts_search_builds_coord_statement(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _Conn("'budget' & 'sam'")

    @contextmanager
    def fake_connection() -> Any:
        yield conn

    monkeypatch.setattr(postgres, "connection", fake_connection)
    postgres.fts_search("budget sam", 7, source="a.md")
    statement, params = conn.executed[-1]
    assert statement.count("THEN 1 ELSE 0 END") == 2
    assert "ts_rank_cd" in statement and ", 32)" in statement
    assert "ORDER BY score DESC, id ASC" in statement and "source_file = %s" in statement
    assert statement.count("%s") == len(params)
    assert "@example.com" not in statement and "[A-Za-z0-9._%+-]+@" not in statement
    assert params[0] == postgres._EMAIL_REGEX
    assert params[-2:] == ("a.md", 7)
    assert "email address contact" in statement
    assert "email" not in params[1]


def test_fts_search_falls_back_to_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _Conn("!'foo'")
    postgres.fts_search("-foo", conn=conn)
    statement, params = conn.executed[-1]
    assert "' | '" in statement and params == ("-foo", 20)
