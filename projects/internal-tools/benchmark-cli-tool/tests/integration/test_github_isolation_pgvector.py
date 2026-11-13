"""github_chunks rows are invisible to every unfiltered baseline helper (local pgvector only)."""

from __future__ import annotations

import hashlib
from urllib.parse import urlparse

import pytest

from benchmark_cli import retrieval
from benchmark_cli.config import DATABASE_URL, EMBEDDING_DIMENSIONS
from benchmark_cli.storage.postgres import (
    board_url_chunk_rows,
    connection,
    fts_search,
    github_chunk_count,
    initialize_database,
    read_doc,
    replace_github_chunks,
    upsert_chunks,
)

pytestmark = pytest.mark.integration

_REPO = "example/repo"


def _local() -> bool:
    if not DATABASE_URL:
        return False
    parsed = urlparse(DATABASE_URL)
    return parsed.hostname in {"localhost", "127.0.0.1"} and parsed.path.rstrip("/") == "/benchmark_cli"


def _vector(text: str) -> list[float]:
    block = hashlib.sha256(text.encode()).digest()
    values: list[float] = []
    while len(values) < EMBEDDING_DIMENSIONS:
        block = hashlib.sha256(block).digest()
        values.extend(byte / 255.0 for byte in block)
    return values[:EMBEDDING_DIMENSIONS]


def _document(text: str, source: str) -> dict[str, object]:
    return {
        "source_file": source,
        "chunk_index": 0,
        "chunk_text": text,
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "token_count": 20,
        "embedding": _vector(text),
        "metadata": {"source_type": "kb_page", "date": "2026-11-12", "section": "Team"},
    }


def _snapshot() -> dict[str, object]:
    decision = retrieval.retrieve_evidence("core team board", queries=["core team board"])
    first = decision.results[0].chunk.id if decision.results else None
    context = None if first is None else read_doc(first)
    return {
        "phrase": [row.id for row in fts_search("core team", limit=5)],
        "board": board_url_chunk_rows(),
        "search_docs": [(item.chunk.id, item.rrf_score) for item in decision.results],
        "read_doc": None if context is None else [chunk.id for chunk in context.chunks],
    }


def test_github_rows_do_not_change_baseline_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _local():
        pytest.skip("Test only runs against local Postgres database benchmark_cli.")
    initialize_database()
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute("TRUNCATE document_chunks, github_chunks RESTART IDENTITY")
        cursor.execute("ALTER SEQUENCE github_chunks_id_seq RESTART WITH 1000000000")
        conn.commit()
    monkeypatch.setattr(retrieval, "embed_texts", lambda texts: [_vector(text) for text in texts])
    upsert_chunks(
        [
            _document("The core team meets weekly. Board: https://trello.com/b/abc/board", "kb/team.md"),
            _document("Core team members are listed on the portal.", "kb/members.md"),
        ]
    )
    before = _snapshot()
    text = "README.md:L1-L2\nThe core team board is https://trello.com/b/abc/board and the core team owns it."
    replace_github_chunks(
        _REPO,
        "README.md",
        [
            {
                "kind": "doc",
                "chunk_index": 0,
                "symbol": "",
                "header_words": "readme md core team",
                "start_line": 1,
                "end_line": 2,
                "chunk_text": text,
                "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "token_count": 20,
                "embedding": _vector(text),
                "metadata": {"url": ""},
            }
        ],
    )
    assert github_chunk_count() == 1
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute("SELECT min(id) FROM github_chunks")
        assert cursor.fetchone()[0] >= 1_000_000_000
    assert _snapshot() == before
