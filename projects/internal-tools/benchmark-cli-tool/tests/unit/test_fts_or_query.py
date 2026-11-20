"""Legacy OR query remains the fallback when the query yields no lexeme groups."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from benchmark_cli.storage import postgres


@pytest.mark.unit
def test_fts_search_converts_websearch_conjunctions_to_or(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class Cursor:
        def __enter__(self) -> Cursor:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def execute(self, statement: str, parameters: tuple[object, ...]) -> None:
            captured["statement"] = statement
            captured["parameters"] = parameters

        def fetchone(self) -> None:
            return None

        def fetchall(self) -> list[tuple[object, ...]]:
            return []

    class Connection:
        def cursor(self) -> Cursor:
            return Cursor()

    @contextmanager
    def fake_connection() -> object:
        yield Connection()

    monkeypatch.setattr(postgres, "connection", fake_connection)
    assert postgres.fts_search("full sentence query") == []
    statement = str(captured["statement"])
    assert "websearch_to_tsquery('english', %s)::text" in statement
    assert "' & '" in statement and "' | '" in statement
    assert captured["parameters"] == ("full sentence query", 20)
