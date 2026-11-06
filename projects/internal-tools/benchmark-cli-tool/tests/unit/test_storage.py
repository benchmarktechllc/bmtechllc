"""Offline contracts for PostgreSQL persistence helpers."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from pgvector import Vector

from benchmark_cli.storage import postgres


@pytest.mark.unit
def test_existing_embeddings_converts_pgvector_vector_to_plain_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCursor:
        def __enter__(self) -> FakeCursor:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def execute(self, statement: str, parameters: tuple[list[str]]) -> None:
            assert "embedding" in statement
            assert parameters == (["a" * 64],)

        def fetchall(self) -> list[tuple[str, Vector]]:
            return [("a" * 64, Vector([0.1, 0.2]))]

    class FakeConnection:
        def cursor(self) -> FakeCursor:
            return FakeCursor()

    @contextmanager
    def fake_connection():
        yield FakeConnection()

    monkeypatch.setattr(postgres, "connection", fake_connection)

    embeddings = postgres.existing_embeddings(["a" * 64])
    assert embeddings.keys() == {"a" * 64}
    assert embeddings["a" * 64] == pytest.approx([0.1, 0.2])
