"""Indexed Trello list answers against an isolated local pgvector database."""

from __future__ import annotations

import hashlib
from urllib.parse import urlparse

import pytest

from benchmark_cli.agent import run_agent
from benchmark_cli.config import DATABASE_URL, EMBEDDING_DIMENSIONS
from benchmark_cli.models import Citation
from benchmark_cli.storage.postgres import connection, initialize_database, trello_card_rows
from benchmark_cli.trello_sync import parse_board, parse_comment, sync_board

pytestmark = pytest.mark.integration

_TOKEN = "ghp_" + ("a" * 36)
_FIRST_ACTIVITY = "2026-11-01T00:00:00.000Z"
_SECOND_ACTIVITY = "2026-11-02T00:00:00.000Z"


def _is_local_benchmark_database() -> bool:
    """True only for this machine's benchmark_cli database."""
    if not DATABASE_URL:
        return False
    parsed = urlparse(DATABASE_URL)
    return parsed.hostname in {"localhost", "127.0.0.1"} and parsed.path.rstrip("/") == "/benchmark_cli"


def _truncate_trello_tables() -> None:
    if not _is_local_benchmark_database():
        raise RuntimeError("Refusing to truncate because DATABASE_URL is not local benchmark_cli.")
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(
            "TRUNCATE document_chunks, trello_source_cards, trello_sync_state RESTART IDENTITY"
        )
        conn.commit()


def _embed(texts: list[str]) -> list[list[float]]:
    """Deterministic offline vectors. No embedding provider is called."""
    vectors: list[list[float]] = []
    for text in texts:
        block = hashlib.sha256(text.encode()).digest()
        values: list[float] = []
        while len(values) < EMBEDDING_DIMENSIONS:
            block = hashlib.sha256(block).digest()
            values.extend(byte / 255.0 for byte in block)
        vectors.append(values[:EMBEDDING_DIMENSIONS])
    return vectors


class _StubTrello:
    """Board and comment fixtures parsed by the real Trello normalizers."""

    def __init__(self, payload: dict[str, object], comments: dict[str, list[dict[str, object]]]) -> None:
        self.payload = payload
        self.comments = comments
        self.fetched: list[str] = []

    def fetch_board(self, board_id: str) -> object:
        self.fetched.append(board_id)
        return parse_board(self.payload)

    def card_comments(self, card_id: str) -> tuple[object, ...]:
        return tuple(
            parsed
            for item in self.comments.get(card_id, [])
            if (parsed := parse_comment(item)) is not None
        )


def _comment(number: int) -> dict[str, object]:
    sentence = (
        f"Comment {number} records catalog progress for card ccc and does not include credentials. "
    )
    return {
        "id": f"cmt-{number}",
        "date": "2026-11-01T12:00:00.000Z",
        "data": {"text": sentence * 80},
        "memberCreator": {"fullName": "Avery"},
    }


def _card(
    card_id: str,
    name: str,
    list_id: str,
    activity: str,
    *,
    description: str = "",
) -> dict[str, object]:
    return {
        "id": card_id,
        "name": name,
        "desc": description,
        "shortUrl": f"https://trello.com/c/{card_id}",
        "idList": list_id,
        "labels": [],
        "dateLastActivity": activity,
    }


def _board(cards: list[dict[str, object]]) -> dict[str, object]:
    return {
        "id": "tIvVZqio",
        "name": "BMTech General Board",
        "shortLink": "tIvVZqio",
        "url": "https://trello.com/b/tIvVZqio/bmtech-general-board",
        "lists": [
            {"id": "L1", "name": "Backlog"},
            {"id": "L2", "name": "Doing"},
            {"id": "L3", "name": "Done"},
        ],
        "cards": cards,
    }


def _chunk_count() -> int:
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM document_chunks")
        return int(cursor.fetchone()[0])


def _assert_citation_row(citation: Citation) -> None:
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT source_file, content_sha256, chunk_text
            FROM document_chunks
            WHERE id = %s
            """,
            (citation.chunk_id,),
        )
        row = cursor.fetchone()
    assert row is not None
    source_file, digest, text = row
    assert citation.source_file == source_file
    assert citation.content_sha256 == str(digest).strip()
    assert citation.quote in text
    assert citation.quote.startswith("Card: ")


@pytest.mark.integration
def test_synced_board_answers_list_membership_without_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sync a stub board into local pgvector and answer from the indexed rows."""
    if not _is_local_benchmark_database():
        pytest.skip("Test only runs against local Postgres database benchmark_cli.")
    monkeypatch.setattr("benchmark_cli.agent.call_llm", lambda **_kwargs: pytest.fail("model"))
    monkeypatch.setattr(
        "benchmark_cli.agent.call_llm_structured", lambda **_kwargs: pytest.fail("model")
    )
    monkeypatch.setattr(
        "benchmark_cli.agent.retrieve_evidence", lambda *_args, **_kwargs: pytest.fail("retrieval")
    )
    monkeypatch.setattr("benchmark_cli.agent.log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr("benchmark_cli.ingestion.pipeline.embed_texts", _embed)

    initialize_database()
    _truncate_trello_tables()
    try:
        comments = [_comment(number) for number in range(6)]
        first = _StubTrello(
            _board(
                [
                    _card("aaa", "Nightly Discord sync", "L1", _FIRST_ACTIVITY),
                    _card(
                        "bbb",
                        "Notion data job",
                        "L2",
                        _FIRST_ACTIVITY,
                        description=f"Secret desc {_TOKEN}",
                    ),
                    _card("ccc", "RAG CLI integration", "L2", _FIRST_ACTIVITY),
                ]
            ),
            {"ccc": comments},
        )
        sync_board(first, "tIvVZqio")
        assert first.fetched == ["tIvVZqio"]
        rows = trello_card_rows()
        assert len(rows) == 3
        assert {row[2] for row in rows} == {"aaa", "bbb", "ccc"}
        assert _chunk_count() > 3

        doing = run_agent("which trello tickets are in Doing?")
        assert doing.status == "answered"
        assert doing.answer.count("Notion data job") == 1
        assert doing.answer.count("RAG CLI integration") == 1
        assert "Nightly Discord sync" not in doing.answer
        assert [citation.quote for citation in doing.citations] == [
            "Card: Notion data job",
            "Card: RAG CLI integration",
        ]
        rendered = doing.model_dump_json()
        assert _TOKEN not in rendered
        assert "Secret desc" not in doing.answer
        for citation in doing.citations:
            _assert_citation_row(citation)

        unknown = run_agent("which trello tickets are in qa?")
        assert unknown.status == "no_relevant_information"
        assert 'named "qa"' in unknown.answer
        assert unknown.citations == []
        for title in ("Nightly Discord sync", "Notion data job", "RAG CLI integration"):
            assert title not in unknown.answer

        second = _StubTrello(
            _board(
                [
                    _card("aaa", "Nightly Discord sync", "L2", _SECOND_ACTIVITY),
                    _card("bbb", "Notion data job", "L3", _SECOND_ACTIVITY),
                ]
            ),
            {},
        )
        sync_board(second, "tIvVZqio")
        progress = run_agent("which tickets are in active in progress state?")
        assert progress.status == "answered"
        assert progress.answer.count("Nightly Discord sync") == 1
        assert "Notion data job" not in progress.answer
        assert "RAG CLI integration" not in progress.answer
        assert [citation.quote for citation in progress.citations] == ["Card: Nightly Discord sync"]
        _assert_citation_row(progress.citations[0])
    finally:
        if _is_local_benchmark_database():
            _truncate_trello_tables()
