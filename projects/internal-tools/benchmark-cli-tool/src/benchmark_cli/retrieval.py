"""Hybrid vector and full-text chunk retrieval using Reciprocal Rank Fusion."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from time import perf_counter

from .config import MIN_COSINE_SIMILARITY, MIN_KEYWORD_OVERLAP
from .providers.openai import embed_texts
from .storage.postgres import (
    SearchChunk,
    SearchFilters,
    fts_search,
    query_was_canceled,
    reuse_connection,
    vector_search,
)

logger = logging.getLogger(__name__)

CANDIDATE_LIMIT = 40
RRF_K = 60
RETRIEVAL_SOURCES = (
    "discord",
    "kb_page",
    "trello",
    "transcript",
    "policy_doc",
    "user_contributed",
)
PER_SOURCE_LIMIT = 4
RETRIEVAL_BUDGET_SECONDS = 8.0
MAX_EXTRA_SEARCHES = 2
WEAK_MIN_COSINE = 0.15


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
_QUESTION_WORD = re.compile(r"\b(?:what|what's|which|who|how|why|when|where)\b", re.IGNORECASE)
_SUBQUESTION_SPLIT = re.compile(r"\s*(?:\?|;|\band\b|&)\s*", re.IGNORECASE)
# Search phrases widen retrieval. Ground phrases are the only ones that can
# prove a paraphrase. A named product such as Fireflies grounds only on its
# own name, never on "meeting transcript".
_QUERY_EXPANSIONS: tuple[tuple[re.Pattern[str], tuple[str, ...], tuple[str, ...]], ...] = (
    (
        re.compile(r"\brag\s+tool\b", re.IGNORECASE),
        ("benchmark-cli-tool", "retrieval-only RAG CLI", "RAG CLI", "cli-tool"),
        ("benchmark-cli-tool", "retrieval-only RAG CLI", "RAG CLI"),
    ),
    (
        re.compile(r"\b(?:kb|knowledge\s+base)\s+portal\b", re.IGNORECASE),
        ("knowledge base portal", "Discord bot", "knowledge base"),
        ("knowledge base portal", "Discord bot", "knowledge base"),
    ),
    (
        re.compile(r"\bfireflies\b", re.IGNORECASE),
        ("Fireflies", "meeting notetaker", "notetaker", "meeting transcript"),
        ("Fireflies",),
    ),
)
_HOW_CUES = ("markdown", "pull request", "token", "form", "auth", "architecture", "browser")


def _matching_phrases(question: str, *, ground: bool) -> list[str]:
    phrases: list[str] = []
    for pattern, search, grounding in _QUERY_EXPANSIONS:
        if pattern.search(question) is None:
            continue
        phrases.extend(grounding if ground else search)
    return phrases


def synonym_phrases_for(question: str) -> list[str]:
    """Phrases that can ground a paraphrase. Named products use their own name."""
    return _matching_phrases(question, ground=True)


def search_phrases_for(question: str) -> list[str]:
    """Extra searches. These do not by themselves prove the answer."""
    return _matching_phrases(question, ground=False)


def chunk_matches_synonym(question: str, text: str) -> bool:
    """True when the chunk uses a mapped name for this paraphrase."""
    folded = text.casefold()
    return any(phrase.casefold() in folded for phrase in synonym_phrases_for(question))


def expand_queries(question: str) -> list[str]:
    """Original question plus synonym searches for paraphrases of known systems."""
    text = " ".join(question.split())
    if not text:
        return []
    queries = [text]
    for extra in search_phrases_for(text):
        queries.append(extra)
        queries.append(f"{text} {extra}")
    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        key = query.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(query)
    return unique


def split_subquestions(question: str) -> list[str]:
    """Split a question that asks two things. A single clause stays intact."""
    text = " ".join(question.split())
    pieces = [piece.strip(" ?.") for piece in _SUBQUESTION_SPLIT.split(text) if piece.strip(" ?.")]
    if len(pieces) < 2:
        return [text] if text else []
    questionish = [piece for piece in pieces if _QUESTION_WORD.search(piece)]
    if len(questionish) >= 2:
        return questionish
    return [text]


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
    # One connection, one query at a time. Parallel searches each opened a
    # connection and tripped the free-tier statement timeout.
    vector_candidates = _safe_search(
        lambda: vector_search(
            query_embedding,
            CANDIDATE_LIMIT,
            speaker=filters.speaker,
            source=filters.source,
            date=filters.date,
            include_synthetic=filters.include_synthetic,
            source_type=source_type,
        )
    )
    fts_candidates = _safe_search(
        lambda: fts_search(
            query,
            CANDIDATE_LIMIT,
            speaker=filters.speaker,
            source=filters.source,
            date=filters.date,
            include_synthetic=filters.include_synthetic,
            source_type=source_type,
        )
    )
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


def _promote_synonym_matches(
    query: str, pooled: list[HybridSearchResult]
) -> list[HybridSearchResult]:
    """Keep chunks that use a mapped name ahead of the source quota.

    The original wording can fill every source slot with unrelated hits.
    The chunks that actually say ``benchmark-cli-tool`` or ``RAG CLI`` would
    then never reach the synonym check.
    """
    phrases = []
    how = re.search(r"\bhow\b", query, re.IGNORECASE) is not None
    for part in split_subquestions(query) or [query]:
        phrases.extend(synonym_phrases_for(part))
    folded = [phrase.casefold() for phrase in phrases]

    def explains(result: HybridSearchResult) -> bool:
        if not how:
            return False
        text = result.chunk.chunk_text.casefold()
        return any(cue in text for cue in _HOW_CUES)

    def matches(result: HybridSearchResult) -> bool:
        text = result.chunk.chunk_text.casefold()
        return any(phrase in text for phrase in folded)

    return (
        [result for result in pooled if explains(result)]
        + [result for result in pooled if matches(result) and not explains(result)]
        + [result for result in pooled if not matches(result) and not explains(result)]
    )


def _safe_search(call: object) -> list[SearchChunk]:
    """Run one arm. A timeout contributes nothing and does not abort retrieval."""
    try:
        found = call()  # type: ignore[operator]
    except Exception as exc:
        if query_was_canceled(exc):
            logger.warning("Retrieval query timed out; keeping results from other queries.")
            return []
        raise
    return list(found)


def _limited_extra_queries(query: str) -> list[str]:
    """A few specific synonym searches. Skip phrases already in the question."""
    if re.search(r"\b(?:link|url)\b", query, re.IGNORECASE) and re.search(
        r"\bportal\b", query, re.IGNORECASE
    ):
        return []
    folded = query.casefold()
    extras = [phrase for phrase in search_phrases_for(query) if phrase.casefold() not in folded]
    extras.sort(key=len, reverse=True)
    chosen: list[str] = []
    for phrase in extras:
        key = phrase.casefold()
        if any(key in kept.casefold() or kept.casefold() in key for kept in chosen):
            continue
        chosen.append(phrase)
        if len(chosen) >= MAX_EXTRA_SEARCHES:
            break
    return chosen


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
    budget_seconds: float = RETRIEVAL_BUDGET_SECONDS,
) -> RetrievalDecision:
    """Accept grounded evidence only when semantic and lexical checks both pass."""
    if not 0.0 <= min_cosine_similarity <= 1.0:
        raise ValueError("min_cosine_similarity must be between 0 and 1.")
    if min_keyword_overlap < 0:
        raise ValueError("min_keyword_overlap must not be negative.")
    if budget_seconds < 0:
        raise ValueError("budget_seconds must not be negative.")
    pooled: list[HybridSearchResult] = []
    per_source = max(PER_SOURCE_LIMIT, 1)
    started = perf_counter()
    searches = [query.strip()] if query.strip() else []
    searches.extend(_limited_extra_queries(query))
    with reuse_connection():
        for index, expanded_query in enumerate(searches):
            if index and perf_counter() - started > budget_seconds:
                logger.info("Retrieval time budget reached; using results gathered so far.")
                break
            try:
                pooled.extend(
                    search_docs(
                        expanded_query,
                        per_source * 3 if index == 0 else per_source,
                        speaker=speaker,
                        source=source,
                        date=date,
                        include_synthetic=include_synthetic,
                    )
                )
            except Exception as exc:
                if not query_was_canceled(exc):
                    raise
                logger.warning("Retrieval query timed out; keeping results from other queries.")
    pooled = _promote_synonym_matches(query, pooled)
    results = tuple(apply_source_quota(pooled, per_source=per_source, limit=max(top_k, per_source * 4)))
    top_cosine = max(
        (result.vector_score for result in results if result.vector_score is not None), default=None
    )
    overlap = meaningful_keyword_overlap(" ".join(searches), results)
    lexical = overlap >= min_keyword_overlap
    semantic = top_cosine is not None and top_cosine >= min_cosine_similarity
    strict = bool(results) and lexical and semantic
    # Two content words at a modest cosine keep a project or topic note.
    # A single shared brand word does not. Named products are checked later.
    weak = (
        bool(results)
        and overlap >= max(min_keyword_overlap, 2)
        and top_cosine is not None
        and top_cosine >= min(min_cosine_similarity, WEAK_MIN_COSINE)
    )
    synonym_hits = tuple(
        result for result in results if chunk_matches_synonym(query, result.chunk.chunk_text)
    )
    accepted = strict or weak or bool(synonym_hits)
    kept = results if strict or weak else synonym_hits
    return RetrievalDecision(
        status="accepted" if accepted else "no_relevant_information",
        results=kept if accepted else (),
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
