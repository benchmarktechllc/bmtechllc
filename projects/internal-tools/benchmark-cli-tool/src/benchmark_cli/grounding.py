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

from .corpus import is_boilerplate_text, is_synthetic_canary
from .ingestion.clean import redact_secrets
from .models import AnswerResponse, Citation
from .portal_urls import portal_home_from_metadata, portal_home_url
from .retrieval import HybridSearchResult

logger = logging.getLogger(__name__)

ANSWER_PATH_REVISION = "2026-11-07-ask-retrieval"
PORTAL_ACCEPTANCE_QUESTIONS = (
    "What is the link to the knowledge base portal?",
    "Where do I open the knowledge base?",
    "What's the URL for our KB?",
)
EMPLOYEE_ACCEPTANCE_QUESTIONS = (
    "How many employees does BMTech currently have?",
    "What is our current headcount?",
    "How many staff do we have?",
)
CORE_TEAM_ACCEPTANCE_QUESTIONS = (
    "Who are the core team members at BMTech?",
    "Who is on the core team?",
)
CHANNEL_ACCEPTANCE_QUESTIONS = (
    "Which Discord channels are synced?",
    "What Discord channels do we ingest?",
)
UNRELATED_ACCEPTANCE_QUESTIONS = (
    "What was the HNSW p99 latency?",
    "How do I add a page to the knowledge base?",
    "Which Discord bot posts article updates?",
)
ACCEPTANCE_QUESTIONS = (
    *PORTAL_ACCEPTANCE_QUESTIONS,
    *EMPLOYEE_ACCEPTANCE_QUESTIONS,
    *CORE_TEAM_ACCEPTANCE_QUESTIONS,
    *CHANNEL_ACCEPTANCE_QUESTIONS,
)

_CORE_TEAM = re.compile(r"\bcore[\s-]+team(?:mates?| members?)?\b", re.IGNORECASE)
_EXPLICIT_EMPLOYEE_COUNT = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+employees?\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(?:not|no|never|without|don'?t|doesn'?t|didn'?t|cannot|can'?t|neither|nor)\b",
    re.IGNORECASE,
)
_COMPANY_SUBJECT = re.compile(r"\b(?:bmtech|benchmarktech|benchmark\s+tech|the company)\b", re.IGNORECASE)
_OTHER_SUBJECT = re.compile(
    r"\b(?:client|customer|vendor|prospect|example|handbook|competitor)\b",
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
    """Match a request for the knowledge-base home link, not a page inside it."""
    text = question.casefold()
    if re.search(r"\b(in|inside|within|from)\s+the\s+(knowledge[ -]?base|portal|kb)\b", text):
        return False
    return re.search(
        r"\b(?:link|url|uri|address|homepage|home page)\b.{0,40}\b(?:portal|knowledge[ -]?base|kb)\b"
        r"|\b(?:portal|knowledge[ -]?base|kb)\b.{0,40}\b(?:link|url|uri|address|homepage|home page)\b"
        r"|\bwhere\s+(?:do|can|should)\s+i\b.{0,40}\b(?:open|find|reach)\b.{0,40}\b(?:portal|knowledge[ -]?base|kb)\b"
        r"|\b(?:open|visit)\s+the\s+(?:portal|knowledge[ -]?base|kb)\b",
        text,
    ) is not None


def asks_for_employee_count(question: str) -> bool:
    """Match a headcount question, not a handbook or policy that mentions employees."""
    text = question.casefold()
    if re.search(r"\b(?:handbook|policy|guide|manual|benefits)\b", text):
        return False
    return re.search(
        r"\b(?:how many|number of|count of)\b.{0,40}\b(?:employees?|staff|headcount)\b"
        r"|\b(?:employee|staff)\s+(?:count|headcount)\b"
        r"|\bheadcount\b"
        r"|\bcurrent\b.{0,30}\b(?:headcount|number of employees|employee count)\b",
        text,
    ) is not None


def asks_for_core_team(question: str) -> bool:
    """Match a request for who is on the core team, not a decision they made."""
    text = question.casefold()
    if re.search(r"\bcore[\s-]+team", text) is None:
        return False
    if re.search(r"\b(?:decide|decided|decision|pricing|said|discuss(?:ed)?)\b", text):
        return False
    return re.search(r"\b(?:who|members?|roster|names?|list)\b", text) is not None


def asks_for_synced_channels(question: str) -> bool:
    """Match a request for the synced-channel catalog, not one channel's decision."""
    text = question.casefold()
    if re.search(r"\b(?:decide|decided|decision|said|discuss(?:ed)?)\b", text):
        return False
    return (
        re.search(r"\bdiscord\b", text) is not None
        and re.search(r"\bchannels?\b", text) is not None
        and re.search(r"\b(?:sync\w*|ingest\w*|index\w*)\b", text) is not None
    )


def credential_refusal() -> AnswerResponse:
    return AnswerResponse(
        status="no_relevant_information",
        answer=_CREDENTIAL_REFUSAL,
        confidence=0.0,
        citations=[],
    )


def channel_provenance_quote(channel_name: str) -> str:
    """Cite the indexed channel name without copying the message body.

    The stored chunk may contain an older announcement URL or a secret. The
    channel list is metadata, so the displayed citation stays a provenance
    label. The chunk row itself is left unchanged.
    """
    name = channel_name.strip().lstrip("#")
    return f"Indexed Discord channel #{name}."


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


def _company_results(results: Sequence[HybridSearchResult]) -> list[HybridSearchResult]:
    """Drop synthetic canaries and bot-boilerplate chunks before quoting them."""
    kept: list[HybridSearchResult] = []
    for result in results:
        if is_synthetic_canary(result.chunk.source_file, result.chunk.metadata):
            continue
        if is_boilerplate_text(result.chunk.chunk_text):
            continue
        kept.append(result)
    return kept


def _company_employee_count(sentence: str) -> bool:
    """Accept only an affirmative count about this company."""
    if not _EXPLICIT_EMPLOYEE_COUNT.search(sentence):
        return False
    if _NEGATION.search(sentence) or _OTHER_SUBJECT.search(sentence):
        return False
    return _COMPANY_SUBJECT.search(sentence) is not None


def _core_team_quote(text: str) -> str | None:
    """Return a sentence that names members. A heading or a denial is not a roster."""
    redacted = redact_secrets(text)
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", redacted):
        cleaned = sentence.strip(" \t#")
        if not cleaned or not _CORE_TEAM.search(cleaned) or cleaned not in redacted:
            continue
        if _NEGATION.search(cleaned):
            continue
        if re.fullmatch(r"core[\s-]+team\s*:?", cleaned, flags=re.IGNORECASE):
            continue
        if re.search(r"\b(?:is|are|includes?|including|consists)\b", cleaned, flags=re.IGNORECASE) is None:
            continue
        names = [
            word
            for word in re.findall(r"\b[A-Z][a-z]{2,}\b", cleaned)
            if word.casefold() not in {"core", "team"}
        ]
        if names:
            return cleaned
    return None


def portal_link_answer(results: Sequence[HybridSearchResult]) -> AnswerResponse:
    """Cite a retrieved knowledge-base page and the deployment's portal home."""
    configured = portal_home_url()
    selected: HybridSearchResult | None = None
    home: str | None = None
    for result in _company_results(results):
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
    for result in _company_results(results):
        redacted = redact_secrets(result.chunk.chunk_text)
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", redacted):
            cleaned = sentence.strip()
            if not cleaned or cleaned not in redacted or not _company_employee_count(cleaned):
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
    """State that no headcount was found. Collaborator names are not a count."""
    del results
    return AnswerResponse(
        status="no_relevant_information",
        answer=_COUNT_UNCERTAINTY,
        confidence=0.0,
        citations=[],
    )


def core_team_answer(results: Sequence[HybridSearchResult]) -> AnswerResponse:
    """Name a core team only when those words appear in retrieved evidence."""
    for result in _company_results(results):
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
