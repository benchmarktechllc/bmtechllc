"""Inspect hybrid, full-text, or vector retrieval results from the command line."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

from .config import PARENT_CONTEXT_MAX_TOKENS
from .ingestion.clean import redact_secrets
from .providers.openai import embed_texts
from .retrieval import HybridSearchResult, search_docs
from .safety import delimit_untrusted_chunk
from .storage.postgres import ParentContext, SearchChunk, fts_search, read_doc, vector_search


def _chunk_payload(chunk: SearchChunk, rank: int, score: float, score_name: str) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": chunk.id,
        "source_file": chunk.source_file,
        "chunk_index": chunk.chunk_index,
        "content_sha256": chunk.content_sha256,
        score_name: score,
        "untrusted": True,
        "text": delimit_untrusted_chunk(redact_secrets(chunk.chunk_text)),
        "metadata": chunk.metadata,
    }


def _hybrid_payload(results: Sequence[HybridSearchResult]) -> list[dict[str, Any]]:
    return [
        _chunk_payload(result.chunk, rank, result.rrf_score, "rrf_score")
        for rank, result in enumerate(results, start=1)
    ]


def _native_payload(results: Sequence[SearchChunk], score_name: str) -> list[dict[str, Any]]:
    return [
        _chunk_payload(result, rank, result.score, score_name)
        for rank, result in enumerate(results, start=1)
    ]


def _parent_context_payload(context: ParentContext) -> dict[str, Any]:
    return {
        "mode": "read_doc",
        "anchor_chunk_id": context.anchor.id,
        "source_file": context.anchor.source_file,
        "section": context.section,
        "section_complete": context.section_complete,
        "token_count": context.token_count,
        "chunks": [
            {
                "chunk_id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "untrusted": True,
                "text": delimit_untrusted_chunk(redact_secrets(chunk.chunk_text)),
                "metadata": chunk.metadata,
                "token_count": chunk.token_count,
            }
            for chunk in context.chunks
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    request = parser.add_mutually_exclusive_group(required=True)
    request.add_argument("--query", help="Text to retrieve relevant chunks for.")
    request.add_argument("--read-doc", type=int, metavar="CHUNK_ID", help="Read context around a retrieved chunk.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return (default: 5).")
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=PARENT_CONTEXT_MAX_TOKENS,
        help=f"Maximum tokens returned by --read-doc (default: {PARENT_CONTEXT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--mode",
        choices=("hybrid", "fts", "vector"),
        default="hybrid",
        help="Retrieval mode to inspect (default: hybrid).",
    )
    parser.add_argument("--speaker", help="Only return chunks that include this speaker.")
    parser.add_argument("--source", help="Only return chunks from this source filename.")
    parser.add_argument("--date", help="Only return chunks covering this ISO-8601 date (YYYY-MM-DD).")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.max_context_tokens < 1:
        parser.error("--max-context-tokens must be at least 1")

    if args.read_doc is not None:
        context = read_doc(args.read_doc, max_tokens=args.max_context_tokens)
        if context is None:
            parser.error(f"Chunk ID not found: {args.read_doc}")
        print(json.dumps(_parent_context_payload(context), indent=2))
        return

    if args.mode == "hybrid":
        payload = _hybrid_payload(
            search_docs(
                args.query,
                args.top_k,
                speaker=args.speaker,
                source=args.source,
                date=args.date,
            )
        )
    elif args.mode == "fts":
        payload = _native_payload(
            fts_search(
                args.query,
                args.top_k,
                speaker=args.speaker,
                source=args.source,
                date=args.date,
            ),
            "fts_score",
        )
    else:
        payload = _native_payload(
            vector_search(
                embed_texts([args.query])[0],
                args.top_k,
                speaker=args.speaker,
                source=args.source,
                date=args.date,
            ),
            "vector_score",
        )
    print(json.dumps({"mode": args.mode, "query": args.query, "results": payload}, indent=2))


if __name__ == "__main__":
    main()
