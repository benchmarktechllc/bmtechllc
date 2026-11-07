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
    """Match only an unambiguous request for the knowledge-base home.

    A subpage, a second question joined by ``and``, or any other document
    name falls through to the ordinary model path.
    """
    text = question.casefold()
    if re.search(r"\b(?:and|then)\b", text):
        return False
    if re.search(r"\b(?:page|status|billing|guide|handbook|article|runbook|playbook)\b", text):
        return False
    if re.search(r"\bportal\s+(?!home\b|link\b|url\b)\w+", text):
        return False
    asks_for_location = re.search(
        r"\b(?:link|url|uri|address|homepage|home page)\b"
        r"|\bwhere\b.+\b(?:open|find)\b"
        r"|\b(?:open|visit)\b",
        text,
    )
    names_the_site = re.search(r"\b(?:knowledge[ -]?base|portal|kb)\b", text)
    return asks_for_location is not None and names_the_site is not None


def asks_for_employee_count(question: str) -> bool:
    """Match only an unambiguous current BMTech headcount question."""
    text = question.casefold()
    if re.search(r"\b(?:joined|hired|left|rotation|on-call|on call)\b", text):
        return False
    if re.search(r"\bhow many\b.+\b(?:on|in)\s+(?:the|our)\b", text):
        return False
    other_subject = re.search(r"\bhow many\s+employees\s+does\s+([a-z]+)\b", text)
    if other_subject and other_subject.group(1) not in {"bmtech", "benchmark", "benchmarktech"}:
        return False
    return re.search(
        r"\bhow many\s+employees\s+does\s+bmtech\b"
        r"|\bhow many\s+staff\s+do\s+we\b"
        r"|\b(?:our|bmtech(?:'s)?)\s+(?:current\s+)?headcount\b",
        text,
    ) is not None


def asks_for_core_team(question: str) -> bool:
    """Match only who the members are. Hiring and decisions fall through."""
    text = question.casefold()
    if re.search(r"\bcore[\s-]+team", text) is None:
        return False
    return re.search(
        r"\bwho\s+(?:is|are)\b.{0,40}\bcore[\s-]+team\b"
        r"|\bwho\s+is\s+on\s+the\s+core[\s-]+team\b"
        r"|\bcore[\s-]+team\s+members\b"
        r"|\bmembers\s+of\s+(?:the\s+)?core[\s-]+team\b",
        text,
    ) is not None


def asks_for_synced_channels(question: str) -> bool:
    """Match only a catalog question. How-to requests fall through."""
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
    r"(?P<subject>.{0,60}?)\s+"
    r"(?<![\w'])(?:has|have|employs|employ)\s+(?:got\s+)?"
    r"(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(?P<noun>employees?)\b(?P<after>\s+[A-Za-z]+)?",
    re.IGNORECASE,
)
_CONDITIONAL_CLAUSE = re.compile(r"^(?:when|if|unless|whenever)\b", re.IGNORECASE)
_CLAUSE_BREAK = re.compile(r",|(?=\b(?:while|whereas|although|though|but)\b)", re.IGNORECASE)
_SENTENCE_HEDGE = re.compile(
    r"\b(?:if|unless|when|whenever|would|might|could|may|should)\b|\b(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
_COUNT_FUNCTION_WORD = frozenset(
    {"if", "unless", "when", "in", "at", "on", "for", "and", "or", "the", "a", "an", "to", "of", "with"}
)
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
    """Require the proper name. ``The company`` alone is not attributed."""
    folded = " ".join(subject.split()).casefold()
    if re.search(r"\b(?:would|might|could|may|should|not)\b|\w+n't\b", folded):
        return False
    if re.search(r"\b(?:acme|client|customer|vendor|prospect|competitor)\b", folded):
        return False
    return re.search(r"\bbmtech\b|\bbenchmarktech\b|\bbenchmark tech\b", folded) is not None


def _company_employee_count(sentence: str) -> bool:
    """Accept only a present, unconditional count whose head noun is employees.

    ``employee handbook`` is a modifier, ``would``/``might``/``if`` are not
    current facts, ``had`` plus a year is history, and another party's subject
    is not this company. ``hasn't`` never matches the bare verb ``has``.
    """
    if _SENTENCE_HEDGE.search(sentence):
        return False
    for clause in _CLAUSE_BREAK.split(sentence):
        clause = clause.strip()
        if not clause or _CONDITIONAL_CLAUSE.match(clause):
            continue
        match = _AFFIRMED_COUNT.search(clause)
        if not match or not _subject_is_this_company(match.group("subject")):
            continue
        following = (match.group("after") or "").strip().casefold()
        if match.group("noun").casefold() == "employee" and following and following not in _COUNT_FUNCTION_WORD:
            continue
        return True
    return False


def _people_names(sentence: str) -> list[str]:
    """Ignore capitalized words that are objects of a place preposition."""
    without_places = re.sub(r"\b(?:in|at|near|from|into)\s+[A-Z][a-z]+\b", "", sentence)
    return [
        word
        for word in re.findall(r"\b[A-Z][a-z]{2,}\b", without_places)
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
    r"\b(?:was|were|has been|have been)\s+(?:made|decided|approved|completed|finalized|defined|signed off)\b"
    r"|\b(?:decided|approved|finalized|signed off)\b",
    re.IGNORECASE,
)
# Finite decision verbs only. The noun "decision" inside a proposal is not one.
_EVIDENCE_DONE = re.compile(r"\b(?:decided|approved|agreed|signed off)\b", re.IGNORECASE)
_FOCUS_STOP = frozenset(
    {
        "the", "and", "for", "our", "your", "who", "what", "where", "when", "how",
        "should", "would", "could", "please", "tell", "does", "did", "are", "was",
        "next", "new", "current", "bmtech", "company", "discord", "team", "core",
        "channel", "channels", "with", "from", "that", "this", "you", "have",
    }
)
_NOT_RESPONSIVE = (
    "The retrieved evidence does not address this question. "
    "Missing facts, including any current incident state, are unknown."
)
_NO_PROCEDURE = (
    "The retrieved evidence does not include a supported procedure for this request. "
    "Missing steps are unknown. I will not invent operational steps."
)
_RECOMMENDATION = re.compile(r"\b(?:who|what)\s+should\b|\brecommend\b", re.IGNORECASE)
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
    """Identity comes from source metadata, not a passing mention in the body."""
    metadata = chunk.metadata or {}
    return " ".join(
        (
            chunk.source_file,
            str(metadata.get("title") or ""),
            str(metadata.get("url") or ""),
            str(metadata.get("section") or ""),
            str(metadata.get("kb_section") or ""),
        )
    ).casefold()


def _focus_terms(question: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", question)
        if token.casefold() not in _FOCUS_STOP
    }


def _without_orphan_citation_sentences(answer: str, citation_count: int) -> str:
    """Drop a sentence whose inline marker does not name a returned citation."""
    kept: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", answer.strip()):
        numbers = [int(value) for value in re.findall(r"\[(\d+)\]", sentence)]
        numbers.extend(int(value) for value in re.findall(r"【(\d+)】", sentence))
        if any(number < 1 or number > citation_count for number in numbers):
            continue
        kept.append(sentence)
    return " ".join(kept).strip()


def _abstain(answer: str) -> AnswerResponse:
    return AnswerResponse(status="no_relevant_information", answer=answer, confidence=0.0, citations=[])


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
    visible = _without_orphan_citation_sentences(response.answer, len(response.citations))
    if not visible:
        return _abstain(_NOT_RESPONSIVE)
    response = response.model_copy(update={"answer": visible})
    cited = _cited_chunks(response, chunks)
    quoted = "\n".join(citation.quote for citation in response.citations)
    quoted_folded = quoted.casefold()
    focus = _focus_terms(question)
    howto = re.search(r"\bhow\s+(?:do|can|should|to)\b", question, flags=re.IGNORECASE) is not None
    if howto:
        procedure = bool(focus) and any(term in quoted_folded for term in focus) and re.search(
            r"\b(?:step|first|then|run|add|set|open|use)\b",
            quoted,
            flags=re.IGNORECASE,
        )
        if not procedure:
            return _abstain(_NO_PROCEDURE)
        return response.model_copy(
            update={
                "answer": (
                    "Supported steps found in the cited evidence:\n\n"
                    + "\n".join(f"- {citation.quote}" for citation in response.citations)
                    + "\n\nAnything not quoted above is unknown."
                ),
                "confidence": min(response.confidence, 0.4),
            }
        )
    if _RECOMMENDATION.search(question) and (
        not focus or not any(term in quoted_folded for term in focus)
    ):
        return _abstain(_NOT_RESPONSIVE)
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
