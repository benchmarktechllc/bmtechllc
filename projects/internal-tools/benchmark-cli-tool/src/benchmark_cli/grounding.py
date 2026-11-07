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
    supported_steps,
)
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
_STATUS_TOKEN = re.compile(
    r"\bstatus\s*[:=]\s*[^\n.]*|\b(?:no_relevant_information|nothing matched)\b",
    re.IGNORECASE,
)


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
    )


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


def _split_sentences(answer: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", answer.strip()) if sentence.strip()]


def _is_comparison(question: str) -> bool:
    return re.search(
        r"\b(?:differ(?:s|ence|ent)?|compar(?:e|ison|ing|ed)|versus|vs\.?|between)\b",
        question,
        flags=re.IGNORECASE,
    ) is not None


def _rendered(response: AnswerResponse, text: str) -> AnswerResponse | None:
    """Ship only an answer that still says something."""
    cleaned = text.strip()
    if not meaningful_text(cleaned):
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
    response = response.model_copy(update={"answer": visible})
    cited = _cited_chunks(response, chunks)
    quoted = "\n".join(citation.quote for citation in response.citations)
    identity = " ".join(_chunk_identity_text(chunk) for chunk in cited)
    if _is_howto(question) and not answers_question(question, quoted, require_all=True):
        return _abstain(_NOT_RESPONSIVE)
    kept: list[str] = []
    unsupported_decision = False
    for sentence in _split_sentences(visible):
        if not meaningful_text(sentence):
            continue
        verdict = assess_claim(sentence, quoted, identity)
        if verdict == "supported" and (
            answers_question(question, sentence) or answers_question(question, quoted)
        ):
            kept.append(sentence)
            continue
        if claim_status(sentence) == "decided":
            unsupported_decision = True
            continue
        partial = supported_steps(sentence, quoted)
        if partial and meaningful_text(partial) and answers_question(question, partial):
            kept.append(partial)
    disclaimer = _identity_gap(question, cited)
    excerpts = _excerpts(response)
    if disclaimer:
        body = excerpts if meaningful_text(excerpts) else ""
        text = disclaimer if not body else f"{disclaimer}\n\n{body}"
        rendered = _rendered(response, text)
        return rendered if rendered is not None else _abstain(_NOT_RESPONSIVE)
    if kept and kept == _split_sentences(visible) and meaningful_text(visible):
        return response
    if kept:
        rendered = _rendered(response, " ".join(kept))
        if rendered is not None:
            return rendered
    if (
        asks_recommendation(question)
        and answers_question(question, quoted)
        and is_not_current_recommendation(quoted)
        and meaningful_text(excerpts)
    ):
        rendered = _rendered(response, f"{_NOT_CURRENT_RECOMMENDATION}\n\n{excerpts}")
        if rendered is not None:
            return rendered
    if (
        unsupported_decision
        and answers_question(question, quoted)
        and claim_status(quoted) != "decided"
        and meaningful_text(excerpts)
    ):
        rendered = _rendered(response, f"{_PLAN_QUALIFICATION}\n\n{excerpts}")
        if rendered is not None:
            return rendered
    if (
        not kept
        and _asks_outcome(question)
        and answers_question(question, quoted)
        and meaningful_text(excerpts)
    ):
        if claim_status(quoted) == "decided":
            rendered = _rendered(response, excerpts)
        else:
            rendered = _rendered(response, f"{_PLAN_QUALIFICATION}\n\n{excerpts}")
        if rendered is not None:
            return rendered
    if _is_comparison(question) and answers_question(question, quoted) and meaningful_text(excerpts):
        rendered = _rendered(response, excerpts)
        if rendered is not None:
            return rendered
    if _page_is_identified(question, cited) and meaningful_text(excerpts):
        rendered = _rendered(response, excerpts)
        if rendered is not None:
            return rendered
    if _is_howto(question) and answers_question(question, quoted, require_all=True) and meaningful_text(excerpts):
        rendered = _rendered(response, excerpts)
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
        if not answers_question(question, result.chunk.chunk_text):
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
    """Drop an internal status dump, including ``status : nothing matched``."""
    if _STATUS_TOKEN.search(answer) is None:
        return answer
    cleaned = _STATUS_TOKEN.sub(" ", answer)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" \n\t-:;,.")
    return cleaned or _NOT_RESPONSIVE


def scrub_response(response: AnswerResponse) -> AnswerResponse:
    """Remove credential material and internal status codes from visible text."""
    answer = _plain_answer(redact_secrets(response.answer))
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
