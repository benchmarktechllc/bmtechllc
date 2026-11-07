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

from .corpus import is_synthetic_canary
from .ingestion.clean import redact_secrets
from .models import AnswerResponse, Citation
from .portal_urls import portal_home_from_metadata, portal_home_url
from .retrieval import HybridSearchResult
from .storage.postgres import SearchChunk

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
    """Match a request for the knowledge-base home, not a page inside it."""
    text = question.casefold()
    if re.search(
        r"\b(?:in|inside|within|from)\s+(?:the|our|your)\s+(?:knowledge[ -]?base|portal|kb)\b",
        text,
    ):
        return False
    if re.search(r"\b(?:url|link|address)\s+for\s+the\s+(?!portal\b|knowledge|kb\b)\w+", text):
        return False
    return re.search(
        r"\b(?:link|url|uri|address|homepage|home page)\b.{0,40}\b(?:portal|knowledge[ -]?base|kb)\b"
        r"|\b(?:portal|knowledge[ -]?base)\s+(?:link|url|address|homepage)\b"
        r"|\bwhere\s+(?:do|can|should)\s+i\b.{0,40}\b(?:open|find|reach)\b.{0,40}\b(?:portal|knowledge[ -]?base)\b"
        r"|\b(?:open|visit)\s+the\s+(?:portal|knowledge[ -]?base)\b",
        text,
    ) is not None


def asks_for_employee_count(question: str) -> bool:
    """Match a company headcount question, not a count of a rotation or team."""
    text = question.casefold()
    if re.search(r"\bhow many\b.+\b(?:on|in|for)\s+the\b", text):
        return False
    if re.search(r"\bheadcount\b", text):
        return True
    return re.search(
        r"\bhow many\s+(?:employees?|staff)\b"
        r"|\bnumber of\s+(?:employees?|staff)\b"
        r"|\bemployee\s+count\b",
        text,
    ) is not None


def asks_for_core_team(question: str) -> bool:
    """Match who the members are, not an action the core team took."""
    text = question.casefold()
    if re.search(r"\bcore[\s-]+team", text) is None:
        return False
    return re.search(
        r"\bwho\s+(?:is|are)\b.{0,40}\bcore[\s-]+team\b"
        r"|\bwho\b.{0,30}\b(?:on|in)\s+the\s+core[\s-]+team\b"
        r"|\bcore[\s-]+team\s+members\b"
        r"|\bmembers\s+of\s+(?:the\s+)?core[\s-]+team\b",
        text,
    ) is not None


def asks_for_synced_channels(question: str) -> bool:
    """Match a catalog question, not instructions for indexing one channel."""
    text = question.casefold()
    if re.search(r"\bhow\s+(?:do|can|should|to)\b", text):
        return False
    if re.search(r"\b(?:decide|decided|decision|said|discuss(?:ed)?)\b", text):
        return False
    return (
        re.search(r"\bdiscord\b", text) is not None
        and re.search(r"\bchannels\b", text) is not None
        and re.search(r"\b(?:sync\w*|ingest\w*|index\w*)\b", text) is not None
        and re.search(r"\b(?:which|what|list)\b", text) is not None
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
        kept.append(result)
    return kept


_AFFIRMED_COUNT = re.compile(
    r"(?P<subject>.{0,80}?)\s+"
    r"(?<![\w'])(?:has|have|had|employs|employ)\s+(?:got\s+)?"
    r"(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"employees?\b",
    re.IGNORECASE,
)
_CONDITIONAL_CLAUSE = re.compile(r"^(?:when|if|unless|whenever|should|were|would)\b", re.IGNORECASE)
_CLAUSE_BREAK = re.compile(r",|(?=\b(?:while|whereas|although|though|but)\b)", re.IGNORECASE)
_NON_PERSON = frozenset(
    {
        "the",
        "our",
        "their",
        "its",
        "this",
        "that",
        "these",
        "those",
        "core",
        "team",
        "responsible",
        "ready",
        "meeting",
        "tomorrow",
        "today",
        "there",
        "here",
    }
)


def _subject_is_this_company(subject: str) -> bool:
    folded = " ".join(subject.split()).casefold()
    folded = re.sub(r"^(?:currently|now|still|already|today)\s+", "", folded)
    if re.search(r"\bnot\b|\w+n't\b", folded):
        return False
    if re.search(r"\b(?:acme|client|customer|vendor|prospect|competitor)\b", folded):
        return False
    return re.search(r"\b(?:bmtech|benchmarktech|benchmark tech|the company|our company)\b", folded) is not None


def _company_employee_count(sentence: str) -> bool:
    """Accept a count only when this company is the subject of an affirmative clause.

    Conditional clauses and counts attributed to another party are not a
    current headcount. Contractions such as ``hasn't`` do not match the
    affirmative verb, so they are not a separate word list.
    """
    for clause in _CLAUSE_BREAK.split(sentence):
        clause = clause.strip()
        if not clause or _CONDITIONAL_CLAUSE.match(clause):
            continue
        match = _AFFIRMED_COUNT.search(clause)
        if match and _subject_is_this_company(match.group("subject")):
            return True
    return False


def _people_names(sentence: str) -> list[str]:
    return [
        word
        for word in re.findall(r"\b[A-Z][a-z]{2,}\b", sentence)
        if word.casefold() not in _NON_PERSON
    ]


def _core_team_quote(text: str) -> str | None:
    """Return a sentence that names members. Activity and denials are not a roster."""
    redacted = redact_secrets(text)
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", redacted):
        cleaned = sentence.strip(" \t#")
        if not cleaned or cleaned not in redacted or not _CORE_TEAM.search(cleaned):
            continue
        if re.search(r"\bnot\b|\w+n't\b", cleaned, flags=re.IGNORECASE):
            continue
        if re.search(
            r"\b(?:are|is)\s+(?:on\s+)?the\s+core[\s-]+team\b"
            r"|\bcore[\s-]+team\s+members\s+are\b"
            r"|\b(?:includes?|consists\s+of|named)\b",
            cleaned,
            flags=re.IGNORECASE,
        ) is None:
            continue
        if _people_names(cleaned):
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


_ANSWER_COMPLETION = re.compile(
    r"\b(?:was|were|has been|have been|is|got)\s+(?:decided|approved|completed|finalized|defined|signed off)\b"
    r"|\b(?:decided|approved|finalized|signed off)\b"
    r"|\b(?:mvp|timeline|architecture)\b.{0,40}\b(?:defined|decided|approved|completed)\b"
    r"|\b(?:defined|decided|approved|completed)\b.{0,40}\b(?:mvp|timeline|architecture)\b",
    re.IGNORECASE,
)
_EVIDENCE_DONE = re.compile(
    r"\b(?:approved|signed off|final decision|was decided|were decided|we decided|"
    r"has been completed|was completed|were completed)\b"
    r"|\b(?:mvp|timeline|architecture)\b.{0,40}\b(?:is|was|were|has been|have been)\s+"
    r"(?:defined|decided|approved|set|completed)\b"
    r"|\b(?:defined|decided|approved|completed)\s+the\s+(?:mvp|timeline|architecture)\b",
    re.IGNORECASE,
)
_REQUESTED_DOCS = (
    ("deployment guide", ("deployment guide", "deploy guide")),
    ("runbook", ("runbook", "runbooks")),
    ("playbook", ("playbook", "playbooks")),
    ("employee handbook", ("employee handbook",)),
)
_PLAN_QUALIFICATION = (
    "The retrieved evidence describes planning, a proposal, or to-do work. "
    "It does not explicitly state that this was approved, decided, or completed."
)


def _cited_chunks(response: AnswerResponse, chunks: Sequence[SearchChunk]) -> list[SearchChunk]:
    by_id = {chunk.id: chunk for chunk in chunks}
    return [by_id[citation.chunk_id] for citation in response.citations if citation.chunk_id in by_id]


def _claims_completion(answer: str) -> bool:
    return _ANSWER_COMPLETION.search(answer) is not None


def _evidence_records_completion(text: str) -> bool:
    return _EVIDENCE_DONE.search(text) is not None


def _requested_documents(question: str) -> list[tuple[str, tuple[str, ...]]]:
    text = question.casefold()
    return [item for item in _REQUESTED_DOCS if any(alias in text for alias in item[1])]


def _chunk_identity_text(chunk: SearchChunk) -> str:
    metadata = chunk.metadata or {}
    return " ".join(
        (
            chunk.source_file,
            str(metadata.get("title") or ""),
            str(metadata.get("url") or ""),
            str(metadata.get("section") or ""),
            chunk.chunk_text,
        )
    ).casefold()


def _chunk_label(chunk: SearchChunk) -> str:
    metadata = chunk.metadata or {}
    title = str(metadata.get("title") or "").strip()
    url = str(metadata.get("url") or "").strip()
    if title and url:
        return f"{title} ({url})"
    return title or url or chunk.source_file


def _asserts_document(answer: str, aliases: tuple[str, ...]) -> bool:
    for alias in aliases:
        if re.search(rf"\b{re.escape(alias)}\b", answer, flags=re.IGNORECASE) is None:
            continue
        if re.search(
            rf"\b(?:not|no|isn't|is not|without|closest)\b.{{0,50}}\b{re.escape(alias)}\b",
            answer,
            flags=re.IGNORECASE,
        ):
            return False
        if re.search(
            rf"\b{re.escape(alias)}\b.{{0,50}}\b(?:not found|does not exist|isn't|is not|was not found)\b",
            answer,
            flags=re.IGNORECASE,
        ):
            return False
        return True
    return False


def qualify_grounded_answer(
    question: str,
    response: AnswerResponse,
    chunks: Sequence[SearchChunk],
) -> AnswerResponse:
    """Stop completion claims and document identities the citations do not support."""
    if response.status != "answered" or not response.citations:
        return response
    cited = _cited_chunks(response, chunks)
    quoted = "\n".join(citation.quote for citation in response.citations)
    notes: list[str] = []
    if _claims_completion(response.answer) and not _evidence_records_completion(quoted):
        notes.append(_PLAN_QUALIFICATION)
    for label, aliases in _requested_documents(question):
        if not _asserts_document(response.answer, aliases):
            continue
        if any(any(alias in _chunk_identity_text(chunk) for alias in aliases) for chunk in cited):
            continue
        closest = _chunk_label(cited[0]) if cited else "the cited page"
        notes.append(
            f"No retrieved page is identified as the {label}. "
            f"The closest cited page is {closest}. "
            "That page is not established as the requested document."
        )
        break
    if not notes:
        return response
    excerpts = "\n".join(f"- {citation.quote}" for citation in response.citations)
    return response.model_copy(
        update={
            "answer": "\n\n".join(notes) + "\n\n" + excerpts,
            "confidence": min(response.confidence, 0.4),
        }
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
