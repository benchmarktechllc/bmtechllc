"""Tests for deterministic evidence acceptance rather than nearest-neighbor guesses."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from benchmark_cli import retrieval
from benchmark_cli.retrieval import (
    HybridSearchResult,
    meaningful_keyword_overlap,
    retrieve_evidence,
)
from benchmark_cli.storage.postgres import SearchChunk


def result(chunk_id: int, text: str, vector: float | None, fts: float | None) -> HybridSearchResult:
    return HybridSearchResult(SearchChunk(chunk_id, "source.txt", 0, text, {}, 1.0), 0.1, vector, fts)


def chunk(text: str, score: float, chunk_id: int = 1) -> SearchChunk:
    return SearchChunk(chunk_id, "source.txt", 0, text, {}, score, content_sha256="ab" * 32)


@pytest.mark.unit
def test_keyword_overlap_counts_only_keyword_returned_evidence() -> None:
    candidates = (result(1, "HNSW latency provides eleven milliseconds", 0.9, 1.0),)
    assert meaningful_keyword_overlap("What HNSW latency?", candidates) == 2


def _install_search(
    monkeypatch: pytest.MonkeyPatch,
    *,
    text: str,
    vector_score: float,
    fts_score: float | None,
) -> dict[str, object]:
    seen: dict[str, object] = {"connections": 0, "vector_calls": 0, "embed_inputs": []}

    @contextmanager
    def fake_connection():
        seen["connections"] = int(seen["connections"]) + 1
        conn = object()
        seen["conn"] = conn
        yield conn

    def fake_embed(texts: list[str]) -> list[list[float]]:
        seen["embed_inputs"] = list(texts)
        return [[0.1, 0.2] for _ in texts]

    def fake_vector(_embedding: list[float], _limit: int = 40, *, conn: object, **kwargs: object) -> list[SearchChunk]:
        assert conn is seen["conn"]
        seen["vector_calls"] = int(seen["vector_calls"]) + 1
        seen.setdefault("sources", []).append(kwargs.get("source_type"))
        return [chunk(text, vector_score)]

    def fake_fts(_query: str, _limit: int = 40, *, conn: object, **_kwargs: object) -> list[SearchChunk]:
        assert conn is seen["conn"]
        if fts_score is None:
            return []
        return [chunk(text, fts_score)]

    monkeypatch.setattr(retrieval, "connection", fake_connection)
    monkeypatch.setattr(retrieval, "embed_texts", fake_embed)
    monkeypatch.setattr(retrieval, "vector_search", fake_vector)
    monkeypatch.setattr(retrieval, "fts_search", fake_fts)
    return seen


@pytest.mark.unit
def test_retrieve_evidence_requires_both_similarity_and_keywords(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_search(monkeypatch, text="unrelated content", vector_score=0.91, fts_score=1.0)
    decision = retrieve_evidence("HNSW latency", min_cosine_similarity=0.30, min_keyword_overlap=1)
    assert decision.status == "no_relevant_information"
    assert decision.results == ()

    _install_search(monkeypatch, text="HNSW latency", vector_score=0.91, fts_score=1.0)
    decision = retrieve_evidence("HNSW latency", min_cosine_similarity=0.30, min_keyword_overlap=2)
    assert decision.status == "accepted"
    assert len(decision.results) == 1


@pytest.mark.unit
def test_snapshot_keeps_keyword_hits_below_the_cosine_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_search(monkeypatch, text="HNSW latency", vector_score=0.12, fts_score=1.0)
    kept = retrieve_evidence(
        "HNSW latency",
        min_cosine_similarity=0.30,
        min_keyword_overlap=1,
        snapshot_present=True,
    )
    assert kept.status == "accepted"
    assert len(kept.results) == 1
    assert kept.top_cosine_similarity == 0.12

    _install_search(monkeypatch, text="unrelated content", vector_score=0.12, fts_score=1.0)
    dropped = retrieve_evidence(
        "HNSW latency",
        min_cosine_similarity=0.30,
        min_keyword_overlap=1,
        snapshot_present=True,
    )
    assert dropped.status == "no_relevant_information"
    assert dropped.results == ()


@pytest.mark.unit
def test_rewritten_queries_share_one_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install_search(monkeypatch, text="Fanuel Mengistu owns the kiosk", vector_score=0.2, fts_score=1.0)
    decision = retrieve_evidence(
        "Fanuel Mengistu",
        queries=["Fanuel Mengistu kiosk", "Fanuel Mengistu"],
        snapshot_present=True,
        min_cosine_similarity=0.30,
        min_keyword_overlap=1,
    )
    assert seen["connections"] == 1
    assert int(seen["vector_calls"]) > 2
    assert "discord" in seen["sources"]
    assert seen["embed_inputs"] == ["Fanuel Mengistu kiosk", "Fanuel Mengistu"]
    assert decision.status == "accepted"


@pytest.mark.unit
def test_per_source_floor_keeps_discord_when_trello_fills_the_global_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Discord note that is not in the global top ranks still survives."""
    discord = SearchChunk(
        9,
        "discord-sample.json",
        0,
        "Use Anthropic Opus for high reasoning work.",
        {"source_type": "discord", "channel": "sample"},
        0.42,
        content_sha256="cd" * 32,
    )
    trello = SearchChunk(
        3,
        "trello/board/card.md",
        0,
        "Card: Unrelated portal chore",
        {"source_type": "trello"},
        0.91,
        content_sha256="ab" * 32,
    )

    @contextmanager
    def fake_connection():
        yield object()

    def fake_embed(texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    def fake_vector(_embedding: list[float], _limit: int = 40, *, source_type: str | None = None, **_kwargs: object) -> list[SearchChunk]:
        if source_type == "discord":
            return [discord]
        if source_type is None:
            return [trello]
        return []

    def fake_fts(_query: str, _limit: int = 40, *, source_type: str | None = None, **_kwargs: object) -> list[SearchChunk]:
        if source_type == "discord":
            return [discord]
        if source_type is None:
            return [trello]
        return []

    monkeypatch.setattr(retrieval, "connection", fake_connection)
    monkeypatch.setattr(retrieval, "embed_texts", fake_embed)
    monkeypatch.setattr(retrieval, "vector_search", fake_vector)
    monkeypatch.setattr(retrieval, "fts_search", fake_fts)

    decision = retrieve_evidence(
        "what model should we use for high reasoning work?",
        snapshot_present=True,
    )

    assert decision.status == "accepted"
    assert any(result.chunk.id == discord.id for result in decision.results)
    assert any(
        (result.chunk.metadata or {}).get("source_type") == "discord" for result in decision.results
    )
