"""Grounded answers for company questions the model was treating as missing.

These extractors copy only text that is already in a retrieved chunk, or the
portal host configured for the current deployment. They do not invent a
headcount, a roster, or a Pages host.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .ingestion.clean import redact_secrets
from .models import AnswerResponse, Citation
from .portal_urls import portal_home_from_metadata, portal_home_url
from .retrieval import HybridSearchResult

logger = logging.getLogger(__name__)

ANSWER_PATH_REVISION = "2026-11-07-ask-retrieval"
ACCEPTANCE_QUESTIONS = (
    "What is the link to the knowledge base portal?",
    "How many employees does BMTech currently have?",
    "Who are the core team members at BMTech?",
    "Which Discord channels are synced?",
)

_CORE_TEAM = re.compile(r"\bcore[\s-]+team(?:mates?| members?)?\b", re.IGNORECASE)
_EXPLICIT_EMPLOYEE_COUNT = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+employees?\b",
    re.IGNORECASE,
)
_CREDENTIAL_REQUEST = re.compile(
    r"\b(api[_ -]?key|access[_ -]?token|password|passwd|secret|private key|"
    r"personal access token|github pat|credential)\b|\bgh[pousr]_|\bgithub_pat_",
    re.IGNORECASE,
)
_COUNT_UNCERTAINTY = (
    "The retrieved evidence does not state a current employee count. "
    "A core team, collaborator list, or contributor list is not an employee or payroll count."
)
_NO_CORE_TEAM = (
    "The retrieved evidence does not name a core team. "
    "People mentioned only as collaborators are not labeled here as a core team "
    "or as a complete employee roster."
)
_NO_PORTAL_URL = (
    "The retrieved knowledge-base pages do not include an absolute portal URL."
)
_CREDENTIAL_REFUSAL = "I can't help retrieve or repeat credentials, tokens, keys, or passwords."
_NO_CHANNELS = "The retrieval index does not list any synced Discord channel names."


@dataclass(frozen=True, slots=True)
class SyncedDiscordChannel:
    """One indexed Discord channel and a verbatim excerpt from one of its chunks."""

    channel_name: str
    chunk_id: int
    source_file: str
    content_sha256: str
    quote: str


def asks_for_credentials(question: str) -> bool:
    return bool(_CREDENTIAL_REQUEST.search(question))


def asks_for_portal_link(question: str) -> bool:
    text = question.casefold()
    mentions_portal = "portal" in text or "knowledge base" in text or "knowledge-base" in text
    asks_where = any(word in text for word in ("link", "url", "uri", "address", "where"))
    return mentions_portal and asks_where


def asks_for_employee_count(question: str) -> bool:
    text = question.casefold()
    return "employee" in text and any(
        phrase in text for phrase in ("how many", "number of", "headcount", "count of")
    )


def asks_for_core_team(question: str) -> bool:
    return "core team" in question.casefold()


def asks_for_synced_channels(question: str) -> bool:
    text = question.casefold()
    return "discord" in text and "channel" in text and "sync" in text


def credential_refusal() -> AnswerResponse:
    return AnswerResponse(
        status="no_relevant_information",
        answer=_CREDENTIAL_REFUSAL,
        confidence=0.0,
        citations=[],
    )


def verbatim_excerpt(text: str, *, limit: int = 240) -> str:
    """Return a prefix of redacted text so the quote remains an exact substring."""
    cleaned = redact_secrets(text).strip()
    if not cleaned:
        return ""
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip()


def _citation(result: HybridSearchResult, quote: str) -> Citation | None:
    digest = (result.chunk.content_sha256 or "").strip()
    if not digest or not quote or quote not in redact_secrets(result.chunk.chunk_text):
        return None
    return Citation(
        source_file=result.chunk.source_file,
        chunk_id=result.chunk.id,
        content_sha256=digest,
        quote=quote,
    )


def _core_team_quote(text: str) -> str | None:
    redacted = redact_secrets(text)
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", redacted):
        cleaned = sentence.strip()
        if cleaned and _CORE_TEAM.search(cleaned) and cleaned in redacted:
            return cleaned
    return None


def portal_link_answer(results: Sequence[HybridSearchResult]) -> AnswerResponse:
    """Cite a retrieved knowledge-base page and the deployment's portal home."""
    configured = portal_home_url()
    selected: HybridSearchResult | None = None
    home: str | None = None
    for result in results:
        if result.chunk.metadata.get("source_type") != "kb_page":
            continue
        candidate = portal_home_from_metadata(result.chunk.metadata, configured)
        if candidate is None:
            continue
        selected = result
        home = candidate
        break
    if selected is None or home is None:
        return AnswerResponse(
            status="no_relevant_information",
            answer=_NO_PORTAL_URL,
            confidence=0.0,
            citations=[],
        )
    quote = verbatim_excerpt(selected.chunk.chunk_text)
    citation = _citation(selected, quote)
    if citation is None:
        return AnswerResponse(
            status="no_relevant_information",
            answer=_NO_PORTAL_URL,
            confidence=0.0,
            citations=[],
        )
    return AnswerResponse(
        status="answered",
        answer=f"The knowledge base portal is {home}.",
        confidence=0.8,
        citations=[citation],
    )


def stated_employee_count_answer(results: Sequence[HybridSearchResult]) -> AnswerResponse | None:
    """Quote an explicit employee-count sentence when the index contains one."""
    for result in results:
        redacted = redact_secrets(result.chunk.chunk_text)
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", redacted):
            cleaned = sentence.strip()
            if not cleaned or not _EXPLICIT_EMPLOYEE_COUNT.search(cleaned) or cleaned not in redacted:
                continue
            citation = _citation(result, cleaned)
            if citation is None:
                continue
            return AnswerResponse(
                status="answered",
                answer=f"The evidence states an employee count here:\n\n{cleaned}",
                confidence=0.7,
                citations=[citation],
            )
    return None


def employee_count_answer(results: Sequence[HybridSearchResult]) -> AnswerResponse:
    """Refuse to turn a core-team sentence into a headcount."""
    for result in results:
        quote = _core_team_quote(result.chunk.chunk_text)
        citation = _citation(result, quote or "")
        if quote and citation is not None:
            return AnswerResponse(
                status="answered",
                answer=(
                    f"{_COUNT_UNCERTAINTY}\n\n"
                    "The evidence does identify this, which is not a headcount:\n\n"
                    f"{quote}"
                ),
                confidence=0.6,
                citations=[citation],
            )
    return AnswerResponse(
        status="no_relevant_information",
        answer=_COUNT_UNCERTAINTY,
        confidence=0.0,
        citations=[],
    )


def core_team_answer(results: Sequence[HybridSearchResult]) -> AnswerResponse:
    """Name a core team only when those words appear in retrieved evidence."""
    for result in results:
        quote = _core_team_quote(result.chunk.chunk_text)
        citation = _citation(result, quote or "")
        if quote and citation is not None:
            return AnswerResponse(
                status="answered",
                answer=(
                    "The evidence names this core team. "
                    "That label is not an employee or payroll count.\n\n"
                    f"{quote}"
                ),
                confidence=0.7,
                citations=[citation],
            )
    return AnswerResponse(
        status="no_relevant_information",
        answer=_NO_CORE_TEAM,
        confidence=0.0,
        citations=[],
    )


def synced_channels_answer(channels: Sequence[SyncedDiscordChannel]) -> AnswerResponse:
    """List channel names stored on indexed Discord chunks."""
    citations: list[Citation] = []
    names: list[str] = []
    for channel in channels:
        name = channel.channel_name.strip().lstrip("#")
        quote = channel.quote.strip()
        digest = channel.content_sha256.strip()
        if not name or not quote or not digest:
            continue
        if name not in names:
            names.append(name)
        citations.append(
            Citation(
                source_file=channel.source_file,
                chunk_id=channel.chunk_id,
                content_sha256=digest,
                quote=quote,
            )
        )
    if not names or not citations:
        return AnswerResponse(
            status="no_relevant_information",
            answer=_NO_CHANNELS,
            confidence=0.0,
            citations=[],
        )
    listed = ", ".join(f"#{name}" for name in names)
    return AnswerResponse(
        status="answered",
        answer=f"Synced Discord channels in the retrieval index: {listed}.",
        confidence=0.8,
        citations=citations,
    )


def scrub_response(response: AnswerResponse) -> AnswerResponse:
    """Remove credential material from answer text and citation quotes."""
    answer = redact_secrets(response.answer)
    citations = []
    for citation in response.citations:
        quote = redact_secrets(citation.quote).strip()
        if not quote:
            continue
        citations.append(citation.model_copy(update={"quote": quote}))
    if response.status == "answered" and not citations:
        return AnswerResponse(
            status="no_relevant_information",
            answer=answer or _CREDENTIAL_REFUSAL,
            confidence=0.0,
            citations=[],
        )
    if response.status == "no_relevant_information":
        return AnswerResponse(status="no_relevant_information", answer=answer, confidence=0.0, citations=[])
    return response.model_copy(update={"answer": answer, "citations": citations})


def retrieval_log_fields(results: Sequence[HybridSearchResult]) -> Mapping[str, str]:
    """Identifiers safe to write to runtime logs. No chunk text and no secrets."""
    return {
        "chunk_ids": ",".join(str(result.chunk.id) for result in results),
        "sources": ",".join(result.chunk.source_file for result in results),
        "source_types": ",".join(
            str(result.chunk.metadata.get("source_type") or "") for result in results
        ),
    }
