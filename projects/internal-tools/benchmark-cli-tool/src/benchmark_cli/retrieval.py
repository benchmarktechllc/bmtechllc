"""Hybrid vector and full-text chunk retrieval using Reciprocal Rank Fusion."""

from __future__ import annotations

import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

from .config import MIN_COSINE_SIMILARITY, MIN_KEYWORD_OVERLAP
from .providers.openai import embed_texts
from .storage.postgres import SearchChunk, SearchFilters, fts_search, vector_search

CANDIDATE_LIMIT = 40
RRF_K = 60


@dataclass(frozen=True, slots=True)
class HybridSearchResult:
    """A deduplicated chunk annotated with its fused rank score."""

    chunk: SearchChunk
    rrf_score: float
    vector_score: float | None = None
    fts_score: float | None = None


@dataclass(frozen=True, slots=True)
class RetrievalDecision:
    """Evidence accepted for answering, or an explicit abstention reason."""

    status: str
    results: tuple[HybridSearchResult, ...]
    top_cosine_similarity: float | None
    keyword_overlap: int


_KEYWORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")
_STOPWORDS = frozenset({"a", "an", "and", "are", "as", "at", "by", "for", "from", "how", "in", "is", "of", "on", "or", "the", "to", "what", "when", "which", "who", "with"})


def meaningful_keyword_overlap(query: str, candidates: Sequence[HybridSearchResult]) -> int:
    """Count meaningful query tokens present in the keyword-retrieved evidence."""
    keywords = {
        token.casefold()
        for token in _KEYWORD.findall(query)
        if token.casefold() not in _STOPWORDS and len(token) > 1
    }
    if not keywords:
        return 0
    fts_text = " ".join(
        result.chunk.chunk_text.casefold() for result in candidates if result.fts_score is not None
    )
    return sum(bool(re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", fts_text)) for token in keywords)


def rrf_fuse(
    ranked_lists: Sequence[Sequence[SearchChunk]], *, rrf_k: int = RRF_K
) -> list[HybridSearchResult]:
    """Merge ranked lists with 1-based Reciprocal Rank Fusion scoring."""
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative.")
    scores: dict[int, float] = {}
    chunks: dict[int, SearchChunk] = {}
    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            chunks.setdefault(chunk.id, chunk)
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(
        (HybridSearchResult(chunks[chunk_id], score) for chunk_id, score in scores.items()),
        key=lambda result: (-result.rrf_score, result.chunk.id),
    )


def search_docs(
    query: str,
    top_k: int = 5,
    *,
    speaker: str | None = None,
    source: str | None = None,
    date: str | None = None,
    include_synthetic: bool = False,
) -> list[HybridSearchResult]:
    """Run filtered 20-per-arm keyword/vector retrieval and return RRF top results."""
    if not query.strip():
        return []
    if top_k < 1:
        return []
    filters = SearchFilters(
        speaker=speaker, source=source, date=date, include_synthetic=include_synthetic
    )
    query_embedding = embed_texts([query])[0]
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid-search") as executor:
        vector_future = executor.submit(
            vector_search,
            query_embedding,
            CANDIDATE_LIMIT,
            speaker=filters.speaker,
            source=filters.source,
            date=filters.date,
            include_synthetic=filters.include_synthetic,
        )
        fts_future = executor.submit(
            fts_search,
            query,
            CANDIDATE_LIMIT,
            speaker=filters.speaker,
            source=filters.source,
            date=filters.date,
            include_synthetic=filters.include_synthetic,
        )
        vector_candidates = vector_future.result()
        fts_candidates = fts_future.result()
    vector_scores = {chunk.id: chunk.score for chunk in vector_candidates}
    fts_scores = {chunk.id: chunk.score for chunk in fts_candidates}
    return [
        replace(
            result,
            vector_score=vector_scores.get(result.chunk.id),
            fts_score=fts_scores.get(result.chunk.id),
        )
        for result in rrf_fuse((vector_candidates, fts_candidates))[:top_k]
    ]


def retrieve_evidence(
    query: str,
    top_k: int = 5,
    *,
    speaker: str | None = None,
    source: str | None = None,
    date: str | None = None,
    include_synthetic: bool = False,
    min_cosine_similarity: float = MIN_COSINE_SIMILARITY,
    min_keyword_overlap: int = MIN_KEYWORD_OVERLAP,
) -> RetrievalDecision:
    """Accept grounded evidence only when semantic and lexical checks both pass."""
    if not 0.0 <= min_cosine_similarity <= 1.0:
        raise ValueError("min_cosine_similarity must be between 0 and 1.")
    if min_keyword_overlap < 0:
        raise ValueError("min_keyword_overlap must not be negative.")
    results = tuple(
        search_docs(
            query,
            top_k,
            speaker=speaker,
            source=source,
            date=date,
            include_synthetic=include_synthetic,
        )
    )
    top_cosine = max(
        (result.vector_score for result in results if result.vector_score is not None), default=None
    )
    overlap = meaningful_keyword_overlap(query, results)
    accepted = (
        bool(results)
        and top_cosine is not None
        and top_cosine >= min_cosine_similarity
        and overlap >= min_keyword_overlap
    )
    return RetrievalDecision(
        status="accepted" if accepted else "no_relevant_information",
        results=results if accepted else (),
        top_cosine_similarity=top_cosine,
        keyword_overlap=overlap,
    )


def hybrid_search(
    query: str,
    top_k: int = 5,
    *,
    speaker: str | None = None,
    source: str | None = None,
    date: str | None = None,
    include_synthetic: bool = False,
) -> list[HybridSearchResult]:
    """Backward-compatible name for :func:`search_docs`."""
    return search_docs(
        query,
        top_k,
        speaker=speaker,
        source=source,
        date=date,
        include_synthetic=include_synthetic,
    )


def vector_only_search(
    query: str,
    top_k: int = 5,
    *,
    speaker: str | None = None,
    source: str | None = None,
    date: str | None = None,
    include_synthetic: bool = False,
) -> list[HybridSearchResult]:
    """Return vector-only results in the common evaluation result shape."""
    if not query.strip() or top_k < 1:
        return []
    filters = SearchFilters(
        speaker=speaker, source=source, date=date, include_synthetic=include_synthetic
    )
    candidates = vector_search(
        embed_texts([query])[0],
        top_k,
        speaker=filters.speaker,
        source=filters.source,
        date=filters.date,
        include_synthetic=filters.include_synthetic,
    )
    return [
        HybridSearchResult(chunk, rrf_score=0.0, vector_score=chunk.score)
        for chunk in candidates
    ]
