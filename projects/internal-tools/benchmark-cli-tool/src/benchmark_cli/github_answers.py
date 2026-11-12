"""Answers that come from indexed GitHub files and pull requests."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from .models import AnswerResponse, Citation
from .storage.postgres import SearchChunk

_RECENT_PULLS = re.compile(
    r"\b(?:what|which)\b[\s\S]{0,40}\bchanged\b[\s\S]{0,40}\b(?:prs?|pull requests?|commits?|merges?)\b"
    r"|\b(?:last|recent|latest)\b[\s\S]{0,40}\b(?:prs?|pull requests?|commits?|merges?)\b",
    re.IGNORECASE,
)


def asks_for_recent_pulls(question: str) -> bool:
    """True only for a question about recent pull requests, commits, or merges."""
    return _RECENT_PULLS.search(question) is not None


def recent_pulls_answer(chunks: Sequence[SearchChunk], *, limit: int = 5) -> AnswerResponse | None:
    """Newest indexed pull requests. Embeddings do not order by time."""
    rows = [
        chunk
        for chunk in chunks
        if str((chunk.metadata or {}).get("kind") or "") == "pull_request"
        or "/pulls/" in chunk.source_file
    ]
    if not rows:
        return AnswerResponse(
            status="no_relevant_information",
            answer="No indexed pull requests are available.",
            confidence=0.0,
            citations=[],
        )
    ordered = sorted(rows, key=lambda chunk: str((chunk.metadata or {}).get("merged_at") or ""), reverse=True)
    chosen = ordered[:limit]
    lines = ["Recent pull requests:"]
    citations: list[Citation] = []
    for chunk in chosen:
        title = _title_line(chunk.chunk_text)
        url = str((chunk.metadata or {}).get("url") or "")
        if url and url not in chunk.chunk_text:
            url = ""
        when = str((chunk.metadata or {}).get("merged_at") or "")[:10]
        suffix = f" (merged {when})" if when else ""
        link = f" {url}" if url else ""
        lines.append(f"- {title}{suffix}{link}".rstrip())
        quote = title if title in chunk.chunk_text else chunk.chunk_text.strip()
        digest = (chunk.content_sha256 or "").strip()
        if digest and quote:
            citations.append(
                Citation(
                    source_file=chunk.source_file,
                    chunk_id=chunk.id,
                    content_sha256=digest,
                    quote=quote,
                )
            )
    if not citations:
        return None
    return AnswerResponse(status="answered", answer="\n".join(lines), confidence=0.8, citations=citations)


def append_github_links(
    response: AnswerResponse, evidence: Mapping[int, SearchChunk]
) -> AnswerResponse:
    """Add blob links from cited chunks. The model does not invent the URL."""
    if response.status != "answered":
        return response
    links: list[str] = []
    for citation in response.citations:
        chunk = evidence.get(citation.chunk_id)
        if chunk is None:
            continue
        metadata = chunk.metadata or {}
        if str(metadata.get("source_type") or "") != "github":
            continue
        url = str(metadata.get("url") or "")
        path = str(metadata.get("path") or citation.source_file)
        if not url.startswith("https://github.com/") or url not in chunk.chunk_text:
            continue
        if url in response.answer:
            continue
        links.append(f"- [{path}]({url})")
    if not links:
        return response
    answer = response.answer.rstrip() + "\n\n" + "\n".join(links)
    return response.model_copy(update={"answer": answer})


def _title_line(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("Pull request #"):
            return line.removeprefix("Pull request ").strip()
    return text.splitlines()[0].strip() if text.strip() else "Pull request"
