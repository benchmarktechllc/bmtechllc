"""Unit tests for deterministic Reciprocal Rank Fusion behavior."""

from __future__ import annotations

import pytest

from benchmark_cli import retrieval
from benchmark_cli.retrieval import hybrid_search, rrf_fuse
from benchmark_cli.storage.postgres import SearchChunk


def chunk(chunk_id: int) -> SearchChunk:
    return SearchChunk(chunk_id, "source.txt", chunk_id, "text", {}, 1.0)


@pytest.mark.unit
def test_rrf_rewards_chunks_present_in_both_retrieval_modes() -> None:
    fused = rrf_fuse(([chunk(1), chunk(2)], [chunk(2), chunk(3)]))

    assert [result.chunk.id for result in fused] == [2, 1, 3]
    assert fused[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)


@pytest.mark.unit
def test_rrf_uses_one_based_ranks_and_deduplicates_chunks() -> None:
    fused = rrf_fuse(([chunk(7), chunk(7)],), rrf_k=60)

    assert len(fused) == 1
    assert fused[0].rrf_score == pytest.approx(1 / 61 + 1 / 62)


@pytest.mark.unit
def test_rrf_rejects_negative_constant() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        rrf_fuse(([chunk(1)],), rrf_k=-1)


@pytest.mark.unit
def test_hybrid_search_requests_forty_candidates_from_each_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object, int]] = []
    monkeypatch.setattr(retrieval, "embed_texts", lambda _: [[0.1, 0.2]])

    def fake_vector(vector: object, limit: int, **_: object) -> list[SearchChunk]:
        calls.append(("vector", vector, limit))
        return [chunk(1)]

    def fake_fts(query: str, limit: int, **_: object) -> list[SearchChunk]:
        calls.append(("fts", query, limit))
        return [chunk(2)]

    monkeypatch.setattr(retrieval, "vector_search", fake_vector)
    monkeypatch.setattr(retrieval, "fts_search", fake_fts)

    results = hybrid_search("retrieval query", top_k=1)

    assert [result.chunk.id for result in results] == [1]
    assert results[0].vector_score == 1.0
    assert results[0].fts_score is None
    assert {name for name, _, _ in calls} == {"vector", "fts"}
    assert {limit for _, _, limit in calls} == {40}
