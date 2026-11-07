"""Claim-level entailment for grounded answers.

A citation quote being an exact substring of a chunk is not a reason to show
it. Each answer sentence is a claim with polarity, time, and status. The
claim is supported, unknown, or contradicted by the evidence:

- supported: the evidence establishes the proposition, including a reporting
  paraphrase ("describes" / "covers"), a "says to" restatement of an
  instruction, a pronoun, or a source name that belongs to the citation
- unknown: the evidence does not establish the claim, including when a
  proposal or future note is asked to count as a finished decision
- contradicted: the evidence asserts the opposite of the claim

Status comes from decision and proposal language. A participle such as
"billed" is not a decision. Rendering then refuses an answer that is only
numbering or an off-topic quote.
"""

from __future__ import annotations

import re
from typing import Literal

Verdict = Literal["supported", "unknown", "contradicted"]
Status = Literal["stated", "proposed", "decided"]

_FUNCTION = frozenset(
    """
    a an the of to for in on at by with from into about and or but if unless
    when while our your my me i we you do does did is are was were be been
    being how what where who which whom whose please can could should would
    will may might shall tell give show send there here this that these those
    it its their them than then so not no nor as per via onto upon am have has
    had also just only next new current can many someone please
    """.split()
)
_FRAMING = frozenset(
    "relevant retrieved retrieve excerpt excerpts citation citations source sources following page document evidence text section article chunk".split()
)
_VAGUE = frozenset({"result", "thing", "information", "detail", "anything", "something"})
_REPORT_BASES = (
    "say",
    "describe",
    "cover",
    "explain",
    "note",
    "state",
    "mention",
    "discuss",
    "tell",
    "instruct",
)
_DECISION = re.compile(
    r"\b(?:decided|decides|decide|approved|approves|approve|finalized|finalised|"
    r"signed off|agreed|hired)\b"
    r"|\bdecisions?\s+(?:were|was|are|is\s+)?made\b"
    r"|\bmade\s+(?:the\s+)?decisions?\b"
    r"|\b(?:is|are|was|were)\s+(?:now\s+)?(?:final|complete|completed)\b",
    re.IGNORECASE,
)
_FUTURE = re.compile(
    r"\b(?:will|would|might|could|may|shall|going to)\b",
    re.IGNORECASE,
)
_PROPOSAL = re.compile(
    r"\b(?:proposal|propose|proposed|plan|planning|planned|todo|to-do|draft|"
    r"brainstorm|upcoming|later|next)\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(r"\b(?:not|no|never)\b|\w+n't\b", re.IGNORECASE)
_DEICTIC_REPORT = re.compile(
    r"^\s*(?:(?:the|this|our)\s+(?:page|document|evidence|source|text|section|article|chunk)\s+"
    r"|(?:it|they|this)\s+)"
    r"(?:describes|describe|described|covers|cover|covered|explains|explain|explained|"
    r"notes|note|noted|states|state|stated|says|said|mentions|mentioned)\s+(?:that\s+)?",
    re.IGNORECASE,
)
_SAYS_TO = re.compile(
    r"^\s*(?:(?:the|this|our|a|an)\s+[A-Za-z0-9-]+\s+)?"
    r"(?:says|said|tells|told|instructs|instructed)\s+to\s+",
    re.IGNORECASE,
)
_RECOMMENDATION = re.compile(
    r"\b(?:should\s+we|should\s+i|recommend|who\s+should)\b",
    re.IGNORECASE,
)
_CURRENT_RECOMMENDATION = re.compile(
    r"\b(?:we|i)\s+should\s+hire\b|\brecommend(?:s|ed|ing)?\s+hiring\b",
    re.IGNORECASE,
)
_NOT_CURRENT = re.compile(
    r"\b(?:brainstorm|someday|maybe|might|consider|idea|ideas)\b",
    re.IGNORECASE,
)


def claim_status(text: str) -> Status:
    """Decision and proposal markers. Ordinary -ed words stay stated."""
    future = _FUTURE.search(text) is not None
    decision = _DECISION.search(text) is not None
    if decision and not future:
        return "decided"
    if future or _PROPOSAL.search(text):
        return "proposed"
    return "stated"


def meaningful_text(text: str) -> bool:
    """Reject empty rendering and leftover list numbers such as ``2. 3. 4.``"""
    if re.fullmatch(r"[\d.\s]+", text.strip() or ""):
        return False
    words = re.findall(r"[A-Za-z]{3,}", text)
    return len(words) >= 2


def asks_recommendation(question: str) -> bool:
    return _RECOMMENDATION.search(question) is not None


def is_not_current_recommendation(text: str) -> bool:
    """A brainstorm or hedged note is not a current hiring recommendation."""
    if _CURRENT_RECOMMENDATION.search(text):
        return False
    return _NOT_CURRENT.search(text) is not None


def _lemma(token: str) -> str:
    word = token.casefold().replace("'", "")
    if word in {"said", "told"}:
        return {"said": "say", "told": "tell"}[word]
    if word.endswith("ing") and len(word) - 3 >= 3:
        word = word[:-3]
    elif word.endswith("ed") and len(word) - 2 >= 3:
        word = word[:-2]
    elif word.endswith("es") and len(word) - 2 >= 3:
        word = word[:-2]
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        word = word[:-1]
    return word


def _is_reporting(token: str) -> bool:
    raw = token.casefold()
    if raw in {"said", "told", "says"}:
        return True
    for base in _REPORT_BASES:
        if raw == base or raw in {base + "s", base + "d", base + "ed", base + "es", base + "ing"}:
            return True
        if raw.endswith("ed") and base.startswith(raw[:-2]) and len(base) - len(raw[:-2]) <= 1:
            return True
        if raw.endswith("ing") and len(raw) > 4 and base.startswith(raw[:-3]) and len(base) - len(raw[:-3]) <= 1:
            return True
    return False


def _related(left: str, right: str) -> bool:
    if left == right:
        return True
    short, long = (left, right) if len(left) <= len(right) else (right, left)
    return len(long) - len(short) == 1 and long.startswith(short)


def _covered(term: str, terms: set[str]) -> bool:
    return any(_related(term, other) for other in terms)


def proposition_terms(text: str) -> set[str]:
    """Content of the proposition after attribution and reporting verbs are set aside."""
    stripped = _SAYS_TO.sub("", text.strip(), count=1)
    stripped = _DEICTIC_REPORT.sub("", stripped, count=1)
    terms: set[str] = set()
    for token in re.findall(r"[A-Za-z][A-Za-z'-]{1,}", stripped):
        raw = token.casefold().replace("'", "")
        if raw in _FUNCTION or raw in _FRAMING or _is_reporting(token):
            continue
        lemma = _lemma(token)
        if len(lemma) >= 3 and lemma not in _FUNCTION and lemma not in _FRAMING:
            terms.add(lemma)
    return terms


def _terms_covered(claim: set[str], evidence: set[str]) -> bool:
    return all(_covered(term, evidence) for term in claim)


def _polarity_negated(text: str) -> bool:
    return _NEGATION.search(text) is not None


def _sentences(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    return parts or [text.strip()]


def _contradicts(sentence: str, evidence: str) -> bool:
    claim_terms = proposition_terms(sentence)
    if not claim_terms or not _polarity_negated(sentence) and not any(
        _polarity_negated(part) for part in _sentences(evidence)
    ):
        return False
    claim_neg = _polarity_negated(sentence)
    for part in _sentences(evidence):
        if _polarity_negated(part) == claim_neg:
            continue
        evidence_terms = proposition_terms(part)
        if not evidence_terms:
            continue
        overlap = [term for term in claim_terms if _covered(term, evidence_terms)]
        if overlap and len(overlap) / len(claim_terms) >= 0.5:
            return True
    return False


def _name_in_evidence(name: str, evidence: str) -> bool:
    """Sentence-case plurals match the singular used in the evidence."""
    folded = evidence.casefold()
    raw = name.casefold()
    if raw in folded:
        return True
    singular = raw[:-1] if raw.endswith("s") and len(raw) > 3 else raw
    return singular in folded


def proper_names(sentence: str) -> list[str]:
    return [
        word
        for word in re.findall(r"\b[A-Z][a-z]{2,}\b", sentence)
        if word.casefold() not in _FUNCTION and word.casefold() not in _FRAMING
    ]


def assess_claim(sentence: str, evidence: str, identity: str = "") -> Verdict:
    """Whether this sentence is supported, unknown, or contradicted."""
    if not sentence.strip() or not evidence.strip():
        return "unknown"
    if _contradicts(sentence, evidence):
        return "contradicted"
    if claim_status(sentence) == "decided" and claim_status(evidence) != "decided":
        return "unknown"
    if any(not _name_in_evidence(name, evidence) for name in proper_names(sentence)):
        return "unknown"
    claim_terms = proposition_terms(sentence) - proposition_terms(identity)
    evidence_terms = proposition_terms(evidence)
    if not claim_terms:
        return "supported"
    if _terms_covered(claim_terms, evidence_terms):
        return "supported"
    return "unknown"


def answers_question(question: str, text: str, *, require_all: bool = False) -> bool:
    """The text has to address the question. A contained quote can still be off topic."""
    asked = proposition_terms(question) - _VAGUE
    if not asked:
        return True
    have = proposition_terms(text)
    if require_all:
        return _terms_covered(asked, have)
    return any(_covered(term, have) for term in asked)


def supported_steps(sentence: str, evidence: str) -> str | None:
    """Keep instruction clauses the evidence states, and drop an added step."""
    parts = re.split(r"\s*,?\s*(?:and\s+)?then\s+", sentence.strip(), flags=re.IGNORECASE)
    if len(parts) < 2:
        return None
    evidence_terms = proposition_terms(evidence)
    kept = [
        part.strip(" ,")
        for part in parts
        if proposition_terms(part) and _terms_covered(proposition_terms(part), evidence_terms)
    ]
    if not kept or len(kept) == len(parts):
        return None
    return ", then ".join(kept)
