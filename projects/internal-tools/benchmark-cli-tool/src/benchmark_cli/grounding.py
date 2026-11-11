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
from .entailment import (
    answers_question,
    asks_recommendation,
    assess_claim,
    claim_status,
    is_not_current_recommendation,
    meaningful_text,
    shared_term_count,
    supported_steps,
    touches_question,
)
from .ingestion.clean import redact_secrets
from .models import AnswerResponse, Citation
from .portal_urls import portal_home_from_metadata, portal_home_url
from .retrieval import HybridSearchResult, chunk_matches_synonym
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


# Closed class for "the site itself." Any other content word is a subresource.
_PORTAL_REQUEST = frozenset(
    {
        "what", "whats", "where", "link", "url", "uri", "address", "open", "find",
        "visit", "homepage", "send", "give", "show", "tell", "me", "the", "our",
        "a", "an", "to", "for", "of", "is", "do", "i", "we", "please", "can",
        "you", "my", "us", "there", "here", "in", "on", "at", "by", "with",
        "from", "into", "about", "your", "that", "this", "it", "be", "s",
    }
)
_PORTAL_SITE = frozenset({"portal", "knowledge", "base", "kb", "homepage", "home"})


def asks_for_portal_link(question: str) -> bool:
    """Match only a request whose object is the knowledge-base home.

    Procedure questions and any extra content word (a bridge, settings, a
    status page, a second clause) stay on the ordinary answer path. The
    allowed words are the request and the site, not a list of forbidden pages.
    """
    if re.search(r"\b(?:and|then|how)\b", question, flags=re.IGNORECASE):
        return False
    tokens = re.findall(r"[a-z0-9]+", question.casefold().replace("'", ""))
    asks_location = any(
        token in {"link", "url", "uri", "address", "where", "open", "find", "visit", "homepage"}
        for token in tokens
    )
    names_site = any(token in {"portal", "kb", "knowledge"} for token in tokens)
    if not asks_location or not names_site:
        return False
    extras = [
        token
        for token in tokens
        if token not in _PORTAL_REQUEST
        and token not in _PORTAL_SITE
        and token not in {"bmtech", "benchmark", "benchmarktech"}
    ]
    return extras in ([], ["page"], ["pages"])


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
    """Match a catalog of channels that are synced, ingested, or indexed.

    How-to and decision questions stay on the ordinary path. Discord does not
    have to be named: "what channels has BMTech synced into the KB" is the
    same catalog question.
    """
    text = question.casefold()
    if re.search(r"\bhow\s+(?:do|can|should|to)\b", text):
        return False
    if re.search(r"\b(?:decide|decided|decision|said|discuss(?:ed)?)\b", text):
        return False
    return (
        re.search(r"\bchannels?\b", text) is not None
        and re.search(
            r"\b(?:sync\w*|ingest\w*|index\w*|reads?|reading|listens?|listening|monitors?)\b",
            text,
        )
        is not None
        and re.search(r"\b(?:which|what|list|does)\b", text) is not None
    )


def asks_for_certificate_link(question: str) -> bool:
    """A request for the URL of a certificate, not a how-to about certificates."""
    text = question.casefold()
    if re.search(r"\bhow\s+(?:do|can|should|to)\b", text):
        return False
    return (
        re.search(r"\b(?:link|url)\b", text) is not None
        and re.search(r"\bcertificates?\b", text) is not None
    )


def certificate_url_is_specific(url: str, chunks: Sequence[SearchChunk]) -> bool:
    """The URL's own chunk has to call it a certificate."""
    for chunk in chunks:
        if url in chunk.chunk_text and re.search(r"\bcertificates?\b", chunk.chunk_text, re.IGNORECASE):
            return True
    return False


def certificate_link_answer(
    question: str, chunks: Sequence[SearchChunk]
) -> AnswerResponse | None:
    """Cite a certificate URL only when that chunk is about the certificate.

    A related course can be mentioned. It is not presented as the certificate.
    """
    if not asks_for_certificate_link(question):
        return None
    for chunk in chunks:
        if not chunk.content_sha256 or not re.search(r"\bcertificates?\b", chunk.chunk_text, re.IGNORECASE):
            continue
        urls = re.findall(r"https://[^\s)>\]]+", chunk.chunk_text)
        if not urls:
            continue
        url = urls[0]
        quote = next(
            (
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+|\n+", chunk.chunk_text)
                if url in sentence and sentence.strip() in chunk.chunk_text
            ),
            "",
        )
        if not quote:
            continue
        return AnswerResponse(
            status="answered",
            answer=f"The certificate link is {url}.",
            confidence=0.8,
            citations=[
                Citation(
                    source_file=chunk.source_file,
                    chunk_id=chunk.id,
                    content_sha256=chunk.content_sha256,
                    quote=quote,
                )
            ],
        )
    for chunk in chunks:
        if not chunk.content_sha256:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", chunk.chunk_text):
            cleaned = sentence.strip()
            if not cleaned or cleaned not in chunk.chunk_text or not _related_course(cleaned):
                continue
            return AnswerResponse(
                status="answered",
                answer=f"No certificate link is known. A related course is mentioned: {cleaned}",
                confidence=0.6,
                citations=[
                    Citation(
                        source_file=chunk.source_file,
                        chunk_id=chunk.id,
                        content_sha256=chunk.content_sha256,
                        quote=cleaned,
                    )
                ],
            )
    return AnswerResponse(
        status="no_relevant_information",
        answer="No certificate link is known.",
        confidence=0.0,
        citations=[],
    )


_RELATED_COURSE = re.compile(
    r"\b(?:courses?|lessons?|trainings?|workshops?)\b|\bin\s+\d+\s*min\b",
    re.IGNORECASE,
)
_NAMED_PRODUCTS = ("fireflies", "fathom", "anthropic")
_DEFINITION_SOURCES = frozenset({"kb_page", "discord", "policy_doc", "user_contributed"})
_MODEL_CHOICE = re.compile(
    r"\b(?:what|which)\s+models?\b|\bshould\s+we\s+use\b",
    re.IGNORECASE,
)
_RAG_TOOL = re.compile(r"\brag\s+tool\b", re.IGNORECASE)
_RAG_NAME = re.compile(r"benchmark-cli-tool|\brag cli\b", re.IGNORECASE)
_RAG_DOES = re.compile(
    r"\b(?:is|are|searches|search|answers|answer|retriev\w*|indexes|indexed|cites|cited)\b",
    re.IGNORECASE,
)
_RAG_JOB = re.compile(r"\bingest(?:ion)? jobs?\b", re.IGNORECASE)
_HOW_PORTAL = re.compile(
    r"\bhow\b.+\b(?:(?:kb|knowledge\s+base)\s+portal|portal)\b"
    r"|\b(?:(?:kb|knowledge\s+base)\s+portal|portal)\b.+\bhow\b",
    re.IGNORECASE,
)
_PORTAL_EXPLAIN = re.compile(
    r"\b(?:markdown|forms?|submit(?:ted|s)?|pull request|\bpr\b|tokens?|auth\w*|architecture|browser)\b",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"[.!?…](?:[\"')\]]+)?$")
_COUNT_LINE = re.compile(
    r"^(?:[A-Za-z][A-Za-z0-9 /-]{0,40}\s+\d+)(?:\s*,\s*[A-Za-z][A-Za-z0-9 /-]{0,40}\s+\d+)*$"
)
_BULLET_LINE = re.compile(r"^(?:[-*]|\d+\.)\s+\S")


def _related_course(sentence: str) -> bool:
    """A course title such as ``Agentic AI in 38 min``, not only the word course."""
    return _RELATED_COURSE.search(sentence) is not None


def related_course_quote(chunks: Sequence[SearchChunk]) -> str:
    """A sentence that names a related course, when the evidence has one."""
    for chunk in chunks:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", chunk.chunk_text):
            cleaned = sentence.strip()
            if cleaned and cleaned in chunk.chunk_text and _related_course(cleaned):
                return cleaned
    return ""


def named_products(question: str) -> tuple[str, ...]:
    """Vendor or product decisions. Project words such as kiosk are not products."""
    folded = question.casefold()
    return tuple(name for name in _NAMED_PRODUCTS if re.search(rf"\b{name}\b", folded))


def evidence_names_products(question: str, texts: Sequence[str]) -> bool:
    """Cited evidence has to contain the product the question asked about."""
    names = named_products(question)
    if not names:
        return True
    blob = "\n".join(texts).casefold()
    return all(name in blob for name in names)


def asks_how_portal_works(question: str) -> bool:
    """A how-it-works question, not a request for the portal home URL."""
    return _HOW_PORTAL.search(question) is not None


def link_only_answer(answer: str) -> bool:
    """True when the reply is a URL and a few words, with no explanation."""
    if "http" not in answer.casefold() or _PORTAL_EXPLAIN.search(answer):
        return False
    words = re.findall(r"[a-z0-9']+", re.sub(r"https?://\S+", " ", answer.casefold()))
    return len(words) <= 14


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
        if sentence.strip()
    ]


def line_is_structural(line: str) -> bool:
    """A list row, a count line, or a heading. These are complete on their own."""
    stripped = line.strip()
    if not stripped:
        return True
    if _BULLET_LINE.match(stripped) or _COUNT_LINE.match(stripped) or stripped.endswith(":"):
        return True
    return stripped.startswith("#") or stripped.startswith("http")


def ends_mid_sentence(text: str) -> bool:
    """True when the last visible line stops before a sentence boundary."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    last = lines[-1].strip()
    if line_is_structural(last) or _SENTENCE_END.search(last):
        return False
    if re.search(r"https://\S+$", last):
        return False
    return re.search(r"[.!?]\s+(?:Board|List|Trello cards)\b", last) is None


def close_answer(text: str) -> str:
    """Drop a trailing fragment so the reply does not stop mid-sentence."""
    current = text.rstrip()
    while current and ends_mid_sentence(current):
        lines = current.splitlines()
        last = lines[-1]
        complete = re.findall(r".+?[.!?…](?:[\"')\]]+)?", last)
        if complete:
            lines[-1] = " ".join(sentence.strip() for sentence in complete).strip()
            current = "\n".join(lines).rstrip()
            continue
        lines = lines[:-1]
        while lines and not lines[-1].strip():
            lines.pop()
        current = "\n".join(lines).rstrip()
    return current


def asks_for_rag_tool(question: str) -> bool:
    return _RAG_TOOL.search(question) is not None


def answer_defines_rag_tool(answer: str) -> bool:
    """A definition says what the CLI is or does. A job list does not."""
    if not _RAG_NAME.search(answer) or not _RAG_DOES.search(answer):
        return False
    if _RAG_JOB.search(answer) and not re.search(r"\b(?:is|are)\b", answer, re.IGNORECASE):
        return False
    return True


def _usable_rag_definition(sentence: str) -> bool:
    """A sentence that says what the CLI is. An ingest-job line is not one."""
    folded = sentence.casefold()
    names_tool = _RAG_NAME.search(sentence) is not None or "retrieval-only" in folded
    if not names_tool:
        return False
    ingest = re.search(r"\bingest", folded) is not None
    describes = "retrieval-only" in folded or "rag cli" in folded or _RAG_DOES.search(sentence) is not None
    if ingest and "retrieval-only" not in folded and "rag cli" not in folded:
        return False
    return describes


def rag_answer_quotes_definition(answer: str, chunks: Sequence[SearchChunk]) -> bool:
    """Keep a reply only when it uses a definition, not an ingest-job list."""
    folded = answer.casefold()
    definitions = [
        sentence
        for chunk in chunks
        for sentence in _sentences(chunk.chunk_text)
        if sentence in chunk.chunk_text and _usable_rag_definition(sentence)
    ]
    if not definitions:
        return False
    if "ingest" in folded and "retrieval-only" not in folded and "rag cli" not in folded:
        return False
    return any(sentence.casefold() in folded for sentence in definitions) or (
        "retrieval-only" in folded and "rag cli" in folded
    )


def evidence_explains_portal(chunks: Sequence[SearchChunk]) -> bool:
    return any(_PORTAL_EXPLAIN.search(chunk.chunk_text) for chunk in chunks)


def rag_tool_definition(question: str, chunks: Sequence[SearchChunk]) -> AnswerResponse | None:
    """A short definition of the RAG CLI, not a list of ingest jobs."""
    if not asks_for_rag_tool(question):
        return None
    chosen: list[tuple[SearchChunk, str]] = []
    for chunk in chunks:
        if not chunk.content_sha256:
            continue
        sentences = [sentence for sentence in _sentences(chunk.chunk_text) if sentence in chunk.chunk_text]
        for index, sentence in enumerate(sentences):
            if not _usable_rag_definition(sentence):
                continue
            parts = [sentence]
            if index + 1 < len(sentences):
                follow = sentences[index + 1]
                if _RAG_DOES.search(follow) and "ingest" not in follow.casefold():
                    parts.append(follow)
            quote = " ".join(parts)
            if quote not in chunk.chunk_text:
                quote = sentence
            chosen.append((chunk, quote))
    chosen.sort(
        key=lambda item: (
            "ingest" in item[1].casefold(),
            "retrieval-only" not in item[1].casefold() and "rag cli" not in item[1].casefold(),
        )
    )
    if not chosen:
        mentioned = any(
            _RAG_NAME.search(chunk.chunk_text)
            and (
                _RAG_JOB.search(chunk.chunk_text)
                or str((chunk.metadata or {}).get("source_type") or "") == "trello"
            )
            for chunk in chunks
        )
        if not mentioned:
            return None
        return AnswerResponse(
            status="no_relevant_information",
            answer=(
                "Indexed cards mention this name on ingest jobs, "
                "but they do not define the RAG tool."
            ),
            confidence=0.0,
            citations=[],
        )
    unique: list[tuple[SearchChunk, str]] = []
    seen: set[str] = set()
    for chunk, sentence in chosen:
        key = sentence.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append((chunk, sentence))
        if len(unique) == 2:
            break
    answer = " ".join(sentence for _chunk, sentence in unique)
    return AnswerResponse(
        status="answered",
        answer=answer,
        confidence=0.75,
        citations=[
            Citation(
                source_file=chunk.source_file,
                chunk_id=chunk.id,
                content_sha256=chunk.content_sha256 or "",
                quote=sentence,
            )
            for chunk, sentence in unique
        ],
    )


def portal_how_answer(question: str, chunks: Sequence[SearchChunk]) -> AnswerResponse | None:
    """How the portal works, from the sentences that describe the mechanism."""
    if not asks_how_portal_works(question):
        return None
    chosen: list[tuple[SearchChunk, str]] = []
    for chunk in chunks:
        if not chunk.content_sha256:
            continue
        for sentence in _sentences(chunk.chunk_text):
            if sentence not in chunk.chunk_text or not _PORTAL_EXPLAIN.search(sentence):
                continue
            chosen.append((chunk, sentence))
    if not chosen:
        return None
    unique: list[tuple[SearchChunk, str]] = []
    seen: set[str] = set()
    for chunk, sentence in chosen:
        key = sentence.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append((chunk, sentence))
        if len(unique) == 4:
            break
    return AnswerResponse(
        status="answered",
        answer=" ".join(sentence for _chunk, sentence in unique),
        confidence=0.75,
        citations=[
            Citation(
                source_file=chunk.source_file,
                chunk_id=chunk.id,
                content_sha256=chunk.content_sha256 or "",
                quote=sentence,
            )
            for chunk, sentence in unique
        ],
    )


def model_recommendation_answer(
    question: str, chunks: Sequence[SearchChunk]
) -> AnswerResponse | None:
    """Quote a note that recommends a model. Project topics stay on their own path."""
    if named_products(question) or _MODEL_CHOICE.search(question) is None:
        return None
    skip = {
        "what", "which", "model", "models", "should", "use", "for", "the", "our",
        "work", "high", "when", "with", "does", "need", "needs",
    }
    terms = [
        token
        for token in re.findall(r"[a-z0-9]+", question.casefold())
        if len(token) >= 4 and token not in skip
    ]
    if "reasoning" not in terms and "reasoning" in question.casefold():
        terms.append("reasoning")
    for chunk in chunks:
        if str((chunk.metadata or {}).get("source_type") or "") == "trello" or not chunk.content_sha256:
            continue
        for sentence in _sentences(chunk.chunk_text):
            folded = sentence.casefold()
            if sentence not in chunk.chunk_text:
                continue
            if terms and sum(term in folded for term in terms) < min(2, len(terms)):
                continue
            if not re.search(r"\b(?:opus|model|reasoning)\b", folded):
                continue
            return AnswerResponse(
                status="answered",
                answer=sentence,
                confidence=0.7,
                citations=[
                    Citation(
                        source_file=chunk.source_file,
                        chunk_id=chunk.id,
                        content_sha256=chunk.content_sha256,
                        quote=sentence,
                    )
                ],
            )
    return None


def synonym_evidence_answer(
    question: str, chunks: Sequence[SearchChunk]
) -> AnswerResponse | None:
    """Quote a chunk that uses the indexed name for a paraphrase such as RAG tool."""
    if asks_for_rag_tool(question) or asks_how_portal_works(question):
        return None
    for chunk in chunks:
        if named_products(question) and not evidence_names_products(question, [chunk.chunk_text]):
            continue
        if not chunk.content_sha256 or not chunk_matches_synonym(question, chunk.chunk_text):
            continue
        quote = ""
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", chunk.chunk_text):
            cleaned = sentence.strip()
            if cleaned and cleaned in chunk.chunk_text and chunk_matches_synonym(question, cleaned):
                quote = cleaned
                break
        if not quote and chunk_matches_synonym(question, chunk.chunk_text):
            quote = chunk.chunk_text.strip()
        if not quote or quote not in chunk.chunk_text:
            continue
        return AnswerResponse(
            status="answered",
            answer=quote,
            confidence=0.7,
            citations=[
                Citation(
                    source_file=chunk.source_file,
                    chunk_id=chunk.id,
                    content_sha256=chunk.content_sha256,
                    quote=quote,
                )
            ],
        )
    return None


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
    """A roster entry is a full name or a coordinated list of names.

    One capitalized token is also a place or an ordinary word, so ``Boston``
    alone is not a person. ``in Boston`` is removed before the scan.
    """
    without_places = re.sub(r"\b(?:in|at|near|from|into)\s+[A-Z][a-z]+\b", "", sentence)
    pairs = re.findall(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+\b", without_places)
    listed = re.findall(
        r"\b[A-Z][a-z]{2,}(?:\s*,\s*(?:and\s+)?[A-Z][a-z]{2,})+\b",
        without_places,
    )
    return pairs + listed


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


_NOT_RESPONSIVE = (
    "The retrieved evidence does not address this question. "
    "What it does not say is unknown."
)
_NOT_CURRENT_RECOMMENDATION = (
    "The cited text is not a current recommendation. "
    "It reads as an older note or brainstorm, not a decision about who to hire now."
)
_REQUESTED_DOCS = (
    ("deployment guide", ("deployment guide", "deploy guide")),
    ("runbook", ("runbook", "runbooks")),
    ("playbook", ("playbook", "playbooks")),
    ("employee handbook", ("employee handbook",)),
)
_PLAN_QUALIFICATION = (
    "The retrieved evidence does not explicitly state that this was approved, decided, or completed. "
    "What follows is cited text, and it is not a final decision."
)
# Internal answer-status dumps only. ``status: draft`` in a procedure must stay.
_STATUS_TOKEN = re.compile(
    r"\bstatus\s*[:=]\s*(?:no_relevant_information|nothing matched|answered)\b"
    r"|\b(?:no_relevant_information|nothing matched)\b",
    re.IGNORECASE,
)
_RENDERED_STEPS = re.compile(r"(?m)^\s*\d+[.)]\s+\S")


def _cited_chunks(response: AnswerResponse, chunks: Sequence[SearchChunk]) -> list[SearchChunk]:
    by_id = {chunk.id: chunk for chunk in chunks}
    return [by_id[citation.chunk_id] for citation in response.citations if citation.chunk_id in by_id]


def _requested_documents(question: str) -> list[tuple[str, tuple[str, ...]]]:
    text = question.casefold()
    return [item for item in _REQUESTED_DOCS if any(alias in text for alias in item[1])]


def _norm_phrase(text: str) -> str:
    """Fold hyphens, underscores, and file extensions into words."""
    folded = text.casefold()
    folded = re.sub(r"\.[a-z0-9]{1,8}\b", " ", folded)
    return f" {re.sub(r'[^a-z0-9]+', ' ', folded).strip()} "


def _identity_has(identity: str, alias: str) -> bool:
    norm = _norm_phrase(identity)
    phrase = _norm_phrase(alias).strip()
    return f" {phrase} " in norm or f" {phrase}s " in norm


def _chunk_identity_text(chunk: SearchChunk) -> str:
    """Identity comes from the path, title, and KB page type, not the body."""
    metadata = chunk.metadata or {}
    source_type = str(metadata.get("source_type") or "")
    parts = [
        chunk.source_file,
        str(metadata.get("title") or ""),
        str(metadata.get("url") or ""),
        str(metadata.get("section") or ""),
        str(metadata.get("kb_section") or ""),
    ]
    if source_type == "kb_page":
        # The page type is the knowledge base. "page" stays a body/path word
        # so a different KB page is not treated as an article about pages.
        parts.append("knowledge base kb")
    elif source_type:
        parts.append(source_type.replace("_", " "))
    return " ".join(parts)


def _coverage_text(quote: str, chunk: SearchChunk | None) -> str:
    """Quote plus document identity, so a how-to can be covered by its path."""
    if chunk is None:
        return quote
    return f"{quote}\n{_chunk_identity_text(chunk)}"


def quote_covers_question(question: str, quote: str, chunk: SearchChunk) -> bool:
    """Whether the quote or the cited document's identity answers the question."""
    return answers_question(question, _coverage_text(quote, chunk))


_CONTINUATION_START = re.compile(
    r"^(?:so|and|but|or|nor|because|which|that)\b",
    re.IGNORECASE,
)
_CLOSER_START = "]).,;:}>\"”」"


def starts_mid_sentence(text: str) -> bool:
    """A fragment that begins after the sentence has already started."""
    body = re.sub(r"^(?:[-*]\s+|\d+[.)]\s+)+", "", text.strip()).strip()
    if not body:
        return False
    if body[0] in _CLOSER_START or body[0].islower():
        return True
    return _CONTINUATION_START.match(body) is not None


def has_unbalanced_delimiters(text: str) -> bool:
    """Quotes and brackets that open and never close, or the reverse."""
    pairs = {")": "(", "]": "[", "}": "{", "】": "【", "”": "“"}
    opening = set(pairs.values())
    stack: list[str] = []
    straight = 0
    for char in text:
        if char == '"':
            straight += 1
            continue
        if char in opening:
            stack.append(char)
            continue
        if char in pairs:
            if not stack or stack[-1] != pairs[char]:
                return True
            stack.pop()
    return bool(stack) or straight % 2 == 1


def output_is_rejected(text: str) -> bool:
    """Reject a mid-sentence fragment or an unbalanced quote or bracket."""
    stripped = text.strip()
    if not stripped:
        return False
    if has_unbalanced_delimiters(stripped):
        return True
    first = re.split(r"[\n\r]+", stripped, maxsplit=1)[0]
    return starts_mid_sentence(first)


def _is_howto(question: str) -> bool:
    """A procedure question. A comparison that happens to start with ``how`` is not one."""
    if re.search(
        r"\b(?:differ(?:s|ence|ent)?|compar(?:e|ison|ing|ed)|versus|vs\.?|between)\b",
        question,
        flags=re.IGNORECASE,
    ):
        return False
    return re.search(r"\bhow\s+(?:do|can|should|to)\b", question, flags=re.IGNORECASE) is not None


def _asks_outcome(question: str) -> bool:
    return re.search(
        r"\b(?:decision|decisions|decided|approved|complete|completed|final|finalized|done|hire|hired)\b",
        question,
        flags=re.IGNORECASE,
    ) is not None


def _is_document_lookup(question: str) -> bool:
    """A page lookup, not a comparison or explanation of the ideas."""
    if re.search(
        r"\b(?:differ(?:s|ence|ent)?|compar(?:e|ison|ing|ed)|versus|vs\.?|between|rather)\b",
        question,
        flags=re.IGNORECASE,
    ):
        return False
    return re.search(
        r"\b(?:where|find|link|url|locate|which\s+(?:page|document))\b",
        question,
        flags=re.IGNORECASE,
    ) is not None


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


def _excerpts(response: AnswerResponse) -> str:
    return "\n".join(f"- {citation.quote}" for citation in response.citations)


def _with_note(response: AnswerResponse, note: str) -> AnswerResponse:
    return response.model_copy(
        update={
            "answer": f"{note}\n\n{_excerpts(response)}",
            "confidence": min(response.confidence, 0.4),
        }
    )


def _identity_gap(question: str, cited: Sequence[SearchChunk]) -> str | None:
    """A page lookup whose metadata is a different document. Comparisons are not lookups."""
    if not _is_document_lookup(question):
        return None
    for label, aliases in _requested_documents(question):
        if any(
            any(_identity_has(_chunk_identity_text(chunk), alias) for alias in aliases) for chunk in cited
        ):
            return None
        if not cited:
            return None
        closest = _chunk_label(cited[0])
        return (
            f"No retrieved page is identified as the {label}. "
            f"The closest cited page is {closest}. "
            "That page is not established as the requested document."
        )
    return None


def _page_is_identified(question: str, cited: Sequence[SearchChunk]) -> bool:
    if not _is_document_lookup(question):
        return False
    for _label, aliases in _requested_documents(question):
        if any(
            any(_identity_has(_chunk_identity_text(chunk), alias) for alias in aliases) for chunk in cited
        ):
            return True
    return False


_ABBREVIATION = re.compile(r"\b(?:e\.g|i\.e)\.", re.IGNORECASE)


def _split_sentences(answer: str) -> list[str]:
    """Split on sentence boundaries. ``e.g.`` and ``i.e.`` are not boundaries."""
    protected = _ABBREVIATION.sub(lambda match: match.group(0).replace(".", "\u0000"), answer.strip())
    parts = re.split(r"(?<=[.!?])\s+", protected)
    return [part.replace("\u0000", ".").strip() for part in parts if part.strip()]


def _is_comparison(question: str) -> bool:
    return re.search(
        r"\b(?:differ(?:s|ence|ent)?|compar(?:e|ison|ing|ed)|versus|vs\.?|between)\b",
        question,
        flags=re.IGNORECASE,
    ) is not None


_HEADER_LINE = re.compile(r"^(?:Card|Board|List|URL|Labels|Due):")
_STEP_LINE = re.compile(r"^\d+[.)]\s+\S")


def _strip_list_marker(text: str) -> str:
    """Drop a leading bullet or number so a quote is not rendered as ``- -``."""
    return re.sub(r"^(?:[-*]\s+|\d+[.)]\s+)+", "", text.strip()).strip()


def _present_quote(quote: str) -> str:
    """One readable excerpt. Trello header rows share a line; list markers do not double."""
    headers: list[str] = []
    body: list[str] = []
    for raw in quote.splitlines() or [quote]:
        line = _strip_list_marker(raw)
        if not line:
            continue
        if _HEADER_LINE.match(line):
            headers.append(line)
        else:
            body.append(line)
    parts: list[str] = []
    if headers:
        parts.append(" / ".join(headers))
    if body:
        parts.append(" ".join(body))
    return "\n".join(parts)


def _numbered_steps(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if _STEP_LINE.match(line.strip())]


_CLIPPED_STEP = re.compile(
    r"(?:[:;,]|\be\.g\.?|\bi\.e\.?|\b(?:and|or|the|a|an|to|of|for|with))\s*$",
    re.IGNORECASE,
)
_TABLE_RULE = re.compile(r"^:?-{3,}:?$")


def _table_fields(line: str) -> list[str] | None:
    """Markdown table row as ``Name: value`` fields. A rule row is empty."""
    stripped = line.strip()
    if not stripped.startswith("|"):
        return None
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    if not cells or all(not cell or _TABLE_RULE.fullmatch(cell) for cell in cells):
        return []
    if len(cells) == 1:
        return [cells[0]]
    name, value = cells[0], " ".join(cell for cell in cells[1:] if cell)
    if not name or name.casefold() in {"field", "column", "name"}:
        return []
    return [f"{name}: {value}".rstrip()]


def _blockquote_text(line: str) -> str:
    """Drop ``>`` and ``[!NOTE]`` markers from one line of a callout."""
    text = re.sub(r"^>\s*", "", line.strip())
    return re.sub(r"^\[!NOTE\]\s*", "", text, flags=re.IGNORECASE).strip()


def _render_procedure(text: str) -> str | None:
    """Numbered steps from the full page.

    A blank line ends the step paragraph. A ``> [!NOTE]`` callout becomes its
    own ``Note:`` line. Tables under a step are folded into that step.
    """
    groups: list[dict[str, list[str]]] = []
    current: dict[str, list[str]] | None = None
    paragraph_open = False

    def close() -> None:
        nonlocal current
        if current is not None:
            groups.append(current)
        current = None

    for raw in text.splitlines():
        stripped = raw.strip()
        if _STEP_LINE.match(stripped):
            close()
            current = {"prose": [stripped], "fields": [], "notes": []}
            paragraph_open = True
            continue
        if current is None:
            continue
        if not stripped:
            paragraph_open = False
            continue
        if stripped.startswith(">"):
            note = _blockquote_text(stripped)
            if note:
                current["notes"].append(note)
            continue
        if not paragraph_open:
            continue
        extracted = _table_fields(stripped)
        if extracted is None:
            current["prose"].append(stripped)
        else:
            current["fields"].extend(extracted)
    close()
    if len(groups) < 2:
        return None
    rendered: list[str] = []
    for group in groups:
        step = " ".join(group["prose"]).strip()
        if group["fields"]:
            step = f"{step.rstrip()} {'; '.join(group['fields'])}".strip()
        if not step or _CLIPPED_STEP.search(re.sub(r"^\d+[.)]\s+", "", step)):
            return None
        rendered.append(step)
        if group["notes"]:
            rendered.append("Note: " + " ".join(group["notes"]))
    return "\n".join(rendered)


def _howto_procedure(
    question: str,
    citations: Sequence[Citation],
    by_id: Mapping[int, SearchChunk],
) -> tuple[str, list[Citation]] | None:
    """Full numbered steps from the cited page, not from a shortened quote."""
    best: tuple[str, list[Citation]] | None = None
    best_count = 1
    for citation in citations:
        chunk = by_id.get(citation.chunk_id)
        if chunk is None:
            continue
        rendered = _render_procedure(chunk.chunk_text)
        if rendered is None:
            continue
        if not answers_question(
            question,
            f"{chunk.chunk_text}\n{_chunk_identity_text(chunk)}",
            require_all=True,
        ):
            continue
        count = len(_numbered_steps(chunk.chunk_text))
        if count > best_count:
            best = (rendered, [citation])
            best_count = count
    return best


def _clean_excerpts(citations: Sequence[Citation]) -> str:
    """Short verbatim quotes. Fragments and whole-page dumps are left out."""
    lines: list[str] = []
    for citation in citations:
        kept: list[str] = []
        for sentence in _split_sentences(citation.quote):
            presented = _strip_list_marker(_present_quote(sentence))
            if (
                not presented
                or _header_only(presented)
                or output_is_rejected(presented)
                or not meaningful_text(presented)
            ):
                continue
            kept.append(presented)
            if len(kept) == 2:
                break
        if not kept:
            continue
        lines.append("- " + " ".join(kept))
    return "\n".join(lines)


def _header_only(text: str) -> bool:
    """A Trello provenance line with no description is not an excerpt."""
    parts = [part.strip() for part in text.split(" / ") if part.strip()]
    return bool(parts) and all(
        re.fullmatch(r"(?:Card|Board|List|URL|Labels|Due):[^\n]*", part) for part in parts
    )


def _claim_evidence(citation: Citation, by_id: Mapping[int, SearchChunk]) -> str:
    """Quote plus the cited chunk, so a short quote does not hide the rest of the page."""
    chunk = by_id.get(citation.chunk_id)
    if chunk is None:
        return citation.quote
    return f"{citation.quote}\n{chunk.chunk_text}"


def _has_clipped_step(text: str) -> bool:
    """A numbered step that stops at a colon, ``e.g.``, or another unfinished clause."""
    for piece in re.split(r"(?=(?:^|\n)\s*\d+[.)]\s+)", text):
        piece = piece.strip()
        if not _STEP_LINE.match(piece):
            continue
        body = re.sub(r"^\d+[.)]\s+", "", piece).strip()
        if body and _CLIPPED_STEP.search(body):
            return True
    return False


def _rendered(response: AnswerResponse, text: str) -> AnswerResponse | None:
    """Ship only an answer that still says something."""
    cleaned = text.strip()
    if not meaningful_text(cleaned) or output_is_rejected(cleaned) or _has_clipped_step(cleaned):
        return None
    if cleaned == response.answer.strip():
        return response
    return response.model_copy(update={"answer": cleaned, "confidence": min(response.confidence, 0.4)})


def qualify_grounded_answer(
    question: str,
    response: AnswerResponse,
    chunks: Sequence[SearchChunk],
) -> AnswerResponse:
    """Keep supported, responsive synthesis. Do not ship numbering or an off-topic quote.

    Entailment allows a paraphrase. It does not treat quote containment, a
    shared verb, or an -ed spelling as proof. A page lookup that is not the
    requested document keeps the closest-page qualification. A comparison the
    evidence supports is kept as that comparison.
    """
    if response.status != "answered" or not response.citations:
        return response
    visible = _without_orphan_citation_sentences(response.answer, len(response.citations))
    visible = " ".join(
        sentence for sentence in _split_sentences(visible) if not output_is_rejected(sentence)
    )
    response = response.model_copy(update={"answer": visible})
    cited = _cited_chunks(response, chunks)
    by_id = {chunk.id: chunk for chunk in chunks}
    quoted = "\n".join(citation.quote for citation in response.citations)
    identity = " ".join(_chunk_identity_text(chunk) for chunk in cited)
    coverage = "\n".join(
        _coverage_text(citation.quote, by_id.get(citation.chunk_id)) for citation in response.citations
    )
    if _is_howto(question) and not answers_question(question, coverage, require_all=True):
        return _abstain(_NOT_RESPONSIVE)
    topical = [
        citation
        for citation in response.citations
        if touches_question(question, _coverage_text(citation.quote, by_id.get(citation.chunk_id)))
    ]
    relevant = [
        citation
        for citation in topical
        if answers_question(question, _coverage_text(citation.quote, by_id.get(citation.chunk_id)))
    ]
    if _is_howto(question):
        procedure = _howto_procedure(question, response.citations, by_id)
        if procedure is not None:
            text, procedure_citations = procedure
            rendered = _rendered(
                response.model_copy(update={"citations": procedure_citations}),
                text,
            )
            if rendered is not None:
                return rendered
    kept: list[str] = []
    unsupported_decision = False
    for sentence in _split_sentences(visible):
        if not meaningful_text(sentence):
            continue
        verdict = next(
            (
                assess_claim(sentence, _claim_evidence(citation, by_id), identity)
                for citation in relevant
                if assess_claim(sentence, _claim_evidence(citation, by_id), identity) == "supported"
            ),
            "unknown",
        )
        if verdict == "supported":
            if not output_is_rejected(sentence):
                kept.append(sentence)
            continue
        if claim_status(sentence) == "decided":
            unsupported_decision = True
            continue
        partial = supported_steps(sentence, quoted)
        if (
            partial
            and meaningful_text(partial)
            and answers_question(question, partial)
            and not output_is_rejected(partial)
        ):
            kept.append(partial)
    disclaimer = _identity_gap(question, cited)
    topical_text = "\n".join(f"- {citation.quote}" for citation in topical)
    relevant_text = "\n".join(f"- {citation.quote}" for citation in relevant)
    cited_text = _excerpts(response)
    if disclaimer:
        body = cited_text if meaningful_text(cited_text) else ""
        text = disclaimer if not body else f"{disclaimer}\n\n{body}"
        rendered = _rendered(response, text)
        return rendered if rendered is not None else _abstain(_NOT_RESPONSIVE)
    if kept and kept == _split_sentences(visible) and meaningful_text(visible):
        if relevant and len(relevant) != len(response.citations):
            return response.model_copy(update={"citations": relevant})
        return response
    if kept:
        rendered = _rendered(
            response.model_copy(update={"citations": relevant or response.citations}),
            " ".join(kept),
        )
        if rendered is not None:
            return rendered
    if (
        asks_recommendation(question)
        and touches_question(question, quoted)
        and is_not_current_recommendation(quoted)
        and meaningful_text(topical_text)
    ):
        rendered = _rendered(
            response.model_copy(update={"citations": topical}),
            f"{_NOT_CURRENT_RECOMMENDATION}\n\n{topical_text}",
        )
        if rendered is not None:
            return rendered
    if (
        unsupported_decision
        and touches_question(question, quoted)
        and claim_status(quoted) != "decided"
        and meaningful_text(topical_text)
    ):
        rendered = _rendered(
            response.model_copy(update={"citations": topical}),
            f"{_PLAN_QUALIFICATION}\n\n{topical_text}",
        )
        if rendered is not None:
            return rendered
    if (
        not kept
        and _asks_outcome(question)
        and touches_question(question, quoted)
        and meaningful_text(topical_text)
    ):
        if claim_status(quoted) == "decided":
            rendered = _rendered(response.model_copy(update={"citations": topical}), topical_text)
        else:
            rendered = _rendered(
                response.model_copy(update={"citations": topical}),
                f"{_PLAN_QUALIFICATION}\n\n{topical_text}",
            )
        if rendered is not None:
            return rendered
    if _is_comparison(question) and relevant_text and answers_question(question, relevant_text):
        rendered = _rendered(response.model_copy(update={"citations": relevant}), relevant_text)
        if rendered is not None:
            return rendered
    if _page_is_identified(question, cited) and meaningful_text(relevant_text):
        rendered = _rendered(response.model_copy(update={"citations": relevant}), relevant_text)
        if rendered is not None:
            return rendered
    broad = [
        citation
        for citation in response.citations
        if shared_term_count(question, _coverage_text(citation.quote, by_id.get(citation.chunk_id))) >= 2
    ]
    pool = relevant or ([] if _is_howto(question) else broad)
    relevant_coverage = "\n".join(
        _coverage_text(citation.quote, by_id.get(citation.chunk_id)) for citation in pool
    )
    clean = _clean_excerpts(pool)
    covered = bool(pool) and (
        answers_question(question, relevant_coverage, require_all=_is_howto(question))
        or (not _is_howto(question) and any(
            shared_term_count(question, _coverage_text(citation.quote, by_id.get(citation.chunk_id))) >= 2
            for citation in pool
        ))
    )
    if clean and covered:
        rendered = _rendered(response.model_copy(update={"citations": pool}), clean)
        if rendered is not None:
            return rendered
    return _abstain(_NOT_RESPONSIVE)


def supported_outcome_answer(
    question: str,
    results: Sequence[HybridSearchResult],
) -> AnswerResponse | None:
    """When the model abstains, still show on-topic outcome evidence.

    A recorded decision is quoted as that decision. A plan is quoted with an
    explicit note that it is not final. Off-topic chunks are not dumped.
    """
    if not _asks_outcome(question):
        return None
    citations: list[Citation] = []
    for result in results:
        if not touches_question(question, result.chunk.chunk_text):
            continue
        quote = verbatim_excerpt(result.chunk.chunk_text)
        citation = _citation(result, quote)
        if citation is not None:
            citations.append(citation)
    if not citations:
        return None
    decided = [citation.quote for citation in citations if claim_status(citation.quote) == "decided"]
    if decided:
        answer = "\n".join(f"- {quote}" for quote in decided)
    elif asks_recommendation(question) and is_not_current_recommendation("\n".join(c.quote for c in citations)):
        answer = _NOT_CURRENT_RECOMMENDATION + "\n\n" + "\n".join(f"- {c.quote}" for c in citations)
    else:
        answer = _PLAN_QUALIFICATION + "\n\n" + "\n".join(f"- {citation.quote}" for citation in citations)
    if not meaningful_text(answer):
        return None
    return AnswerResponse(status="answered", answer=answer, confidence=0.4, citations=citations)


def _plain_answer(answer: str) -> str:
    """Drop an internal status dump, including ``status : nothing matched``.

    Rendered steps and excerpts are left alone. ``status: draft`` is page
    content, not an answer-status code.
    """
    if _RENDERED_STEPS.search(answer):
        return answer
    if _STATUS_TOKEN.search(answer) is None:
        return answer
    cleaned = _STATUS_TOKEN.sub(" ", answer)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" \n\t-:;,.")
    return cleaned or _NOT_RESPONSIVE


def scrub_response(response: AnswerResponse) -> AnswerResponse:
    """Remove credential material and internal status codes from visible text."""
    answer = close_answer(_plain_answer(redact_secrets(response.answer)))
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
