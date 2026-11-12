"""Hybrid vector and full-text chunk retrieval using Reciprocal Rank Fusion."""

from __future__ import annotations

import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

from .config import MIN_COSINE_SIMILARITY, MIN_KEYWORD_OVERLAP
from .providers.openai import embed_texts
from .storage.postgres import SearchChunk, SearchFilters, connection, fts_search, vector_search

CANDIDATE_LIMIT = 40
RRF_K = 60
PER_SOURCE_LIMIT = 4
# A global top-40 is mostly Trello. Each source still contributes its own top hits.
RETRIEVAL_SOURCES = (
    "discord",
    "kb_page",
    "trello",
    "transcript",
    "policy_doc",
    "user_contributed",
    "github",
)
SOURCE_ARM_LIMIT = PER_SOURCE_LIMIT


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


def source_type_of(result: HybridSearchResult) -> str:
    """Indexed source family for one hit. Missing metadata is its own bucket."""
    return str((result.chunk.metadata or {}).get("source_type") or "other")


def apply_source_quota(
    ranked: Sequence[HybridSearchResult],
    *,
    per_source: int = PER_SOURCE_LIMIT,
    limit: int = 16,
) -> list[HybridSearchResult]:
    """Keep the top hits from each source so one source cannot fill the context."""
    if per_source < 1 or limit < 1:
        return []
    chosen: list[HybridSearchResult] = []
    overflow: list[HybridSearchResult] = []
    counts: dict[str, int] = {}
    seen: set[int] = set()
    for result in ranked:
        if result.chunk.id in seen:
            continue
        seen.add(result.chunk.id)
        source = source_type_of(result)
        if counts.get(source, 0) < per_source:
            chosen.append(result)
            counts[source] = counts.get(source, 0) + 1
        else:
            overflow.append(result)
        if len(chosen) >= limit:
            return chosen
    for result in overflow:
        if len(chosen) >= limit:
            break
        chosen.append(result)
    return chosen


def context_limit(top_k: int) -> int:
    """Room for every source arm. One new source does not take another's slots."""
    per_source = max(PER_SOURCE_LIMIT, 1)
    return max(top_k, per_source * len(RETRIEVAL_SOURCES))


def search_docs(
    query: str,
    top_k: int = 5,
    *,
    speaker: str | None = None,
    source: str | None = None,
    date: str | None = None,
    include_synthetic: bool = False,
    source_type: str | None = None,
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
            source_type=source_type,
        )
        fts_future = executor.submit(
            fts_search,
            query,
            CANDIDATE_LIMIT,
            speaker=filters.speaker,
            source=filters.source,
            date=filters.date,
            include_synthetic=filters.include_synthetic,
            source_type=source_type,
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


def _search_texts(query: str, queries: Sequence[str] | None) -> list[str]:
    """At most three non-empty search strings. The raw query is always eligible."""
    source = list(queries) if queries else [query]
    texts: list[str] = []
    for item in source:
        text = " ".join(str(item).split())
        if text and text not in texts:
            texts.append(text)
        if len(texts) == 3:
            break
    return texts


def _accept_evidence(
    *,
    has_results: bool,
    top_cosine: float | None,
    overlap: int,
    min_cosine_similarity: float,
    min_keyword_overlap: int,
    snapshot_present: bool,
) -> bool:
    """Keep hits that clear both floors.

    When the board snapshot is already in the prompt, a miss of the cosine
    floor alone does not drop a set that still shares keywords with the query.
    """
    if not has_results:
        return False
    cosine_ok = top_cosine is not None and top_cosine >= min_cosine_similarity
    overlap_ok = overlap >= min_keyword_overlap
    if cosine_ok and overlap_ok:
        return True
    return bool(snapshot_present and overlap_ok and not cosine_ok)


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
    queries: Sequence[str] | None = None,
    snapshot_present: bool = False,
) -> RetrievalDecision:
    """Hybrid search for rewritten queries on a single connection.

    Each query runs one global hybrid search plus a short hybrid search per
    source type. Reciprocal rank fusion then keeps a floor from every source,
    so Discord and the knowledge base are not dropped when Trello fills the
    global top ranks. The raw question is one of the query strings.
    """
    if not 0.0 <= min_cosine_similarity <= 1.0:
        raise ValueError("min_cosine_similarity must be between 0 and 1.")
    if min_keyword_overlap < 0:
        raise ValueError("min_keyword_overlap must not be negative.")
    texts = _search_texts(query, queries)
    if not texts:
        return RetrievalDecision("no_relevant_information", (), None, 0)
    embeddings = embed_texts(texts)
    if len(embeddings) != len(texts):
        raise RuntimeError("Embedding response length did not match the search strings.")
    vector_scores: dict[int, float] = {}
    fts_scores: dict[int, float] = {}
    ranked_lists: list[list[SearchChunk]] = []
    with connection() as conn:
        for text, embedding in zip(texts, embeddings, strict=True):
            vector_hits = vector_search(
                embedding,
                CANDIDATE_LIMIT,
                speaker=speaker,
                source=source,
                date=date,
                include_synthetic=include_synthetic,
                conn=conn,
            )
            fts_hits = fts_search(
                text,
                CANDIDATE_LIMIT,
                speaker=speaker,
                source=source,
                date=date,
                include_synthetic=include_synthetic,
                conn=conn,
            )
            for chunk in vector_hits:
                current = vector_scores.get(chunk.id)
                if current is None or chunk.score > current:
                    vector_scores[chunk.id] = chunk.score
            for chunk in fts_hits:
                current = fts_scores.get(chunk.id)
                if current is None or chunk.score > current:
                    fts_scores[chunk.id] = chunk.score
            ranked_lists.append(vector_hits)
            ranked_lists.append(fts_hits)
            for source_type in RETRIEVAL_SOURCES:
                source_vector = vector_search(
                    embedding,
                    SOURCE_ARM_LIMIT,
                    speaker=speaker,
                    source=source,
                    date=date,
                    include_synthetic=include_synthetic,
                    source_type=source_type,
                    conn=conn,
                )
                source_fts = fts_search(
                    text,
                    SOURCE_ARM_LIMIT,
                    speaker=speaker,
                    source=source,
                    date=date,
                    include_synthetic=include_synthetic,
                    source_type=source_type,
                    conn=conn,
                )
                for chunk in source_vector:
                    current = vector_scores.get(chunk.id)
                    if current is None or chunk.score > current:
                        vector_scores[chunk.id] = chunk.score
                for chunk in source_fts:
                    current = fts_scores.get(chunk.id)
                    if current is None or chunk.score > current:
                        fts_scores[chunk.id] = chunk.score
                ranked_lists.append(source_vector)
                ranked_lists.append(source_fts)
    fused = [
        replace(
            result,
            vector_score=vector_scores.get(result.chunk.id),
            fts_score=fts_scores.get(result.chunk.id),
        )
        for result in rrf_fuse(ranked_lists)
    ]
    per_source = max(PER_SOURCE_LIMIT, 1)
    results = tuple(
        apply_source_quota(fused, per_source=per_source, limit=context_limit(top_k))
    )
    top_cosine = max(
        (result.vector_score for result in results if result.vector_score is not None),
        default=None,
    )
    overlap = meaningful_keyword_overlap(" ".join(texts), results)
    accepted = _accept_evidence(
        has_results=bool(results),
        top_cosine=top_cosine,
        overlap=overlap,
        min_cosine_similarity=min_cosine_similarity,
        min_keyword_overlap=min_keyword_overlap,
        snapshot_present=snapshot_present,
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
