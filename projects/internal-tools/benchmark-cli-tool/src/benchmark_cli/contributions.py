"""Remember an answer a person supplies when the bot cannot find one.

Contributions are embedded into ``document_chunks`` as ``user_contributed``.
They are not written into the git repository.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime

from .ingestion.clean import redact_secrets
from .ingestion.pipeline import content_hash
from .providers.openai import embed_texts
from .storage.postgres import delete_source, upsert_chunks

MAX_CONTRIBUTION_CHARS = 2000
THANKS_REPLY = "Thanks, saved. I'll use this for future questions."
UNKNOWN_INVITE = (
    "I couldn't find this in Discord, Trello, the KB or meetings. "
    "If you know the answer, tap Add answer or reply to this message and I'll remember it."
)
_SOURCE_PREFIX = "user_contributed/"


def prepare_contribution(text: str) -> str:
    """Trim, cap, and redact a contribution. An empty result should not be stored."""
    cleaned = redact_secrets(" ".join(text.split()))
    if len(cleaned) > MAX_CONTRIBUTION_CHARS:
        cleaned = cleaned[: MAX_CONTRIBUTION_CHARS - 1].rstrip() + "…"
    return cleaned.strip()


def contribution_source_file(channel_id: int | str, message_id: int | str) -> str:
    """Stable source path so one contribution can be listed or removed later."""
    channel = re.sub(r"\W", "", str(channel_id)) or "channel"
    message = re.sub(r"\W", "", str(message_id)) or "message"
    return f"{_SOURCE_PREFIX}{channel}/{message}.md"


def contribution_record(
    *,
    question: str,
    text: str,
    author_id: int | str,
    author_name: str,
    contributed_at: str,
    channel_id: int | str,
    message_id: int | str,
    message_url: str,
) -> dict[str, object]:
    """One ``document_chunks`` row. The text is already redacted."""
    body = prepare_contribution(text)
    source = contribution_source_file(channel_id, message_id)
    metadata = {
        "source_type": "user_contributed",
        "question": " ".join(question.split())[:500],
        "author_id": str(author_id),
        "author_name": author_name.strip() or "a teammate",
        "contributed_at": contributed_at,
        "date": contributed_at[:10],
        "channel_id": str(channel_id),
        "message_id": str(message_id),
        "url": message_url,
    }
    chunk_text = (
        f"Question: {metadata['question']}\n"
        f"According to {metadata['author_name']} on {metadata['date']}:\n"
        f"{body}"
    )
    return {
        "source_file": source,
        "chunk_index": 0,
        "chunk_text": chunk_text,
        "content_sha256": content_hash(chunk_text),
        "token_count": max(1, len(chunk_text.split())),
        "metadata": metadata,
    }


def store_contribution(
    *,
    question: str,
    text: str,
    author_id: int | str,
    author_name: str,
    channel_id: int | str,
    message_id: int | str,
    message_url: str,
    contributed_at: str | None = None,
) -> str:
    """Embed and insert one contribution. Returns the source path."""
    prepared = prepare_contribution(text)
    if not prepared or prepared == "[REDACTED]":
        raise ValueError("contribution must contain text other than a secret.")
    when = contributed_at or datetime.now().astimezone().date().isoformat()
    if len(when) > 10 and "T" in when:
        when = when[:10]
    row = contribution_record(
        question=question,
        text=prepared,
        author_id=author_id,
        author_name=author_name,
        contributed_at=when,
        channel_id=channel_id,
        message_id=message_id,
        message_url=message_url,
    )
    vector = embed_texts([str(row["chunk_text"])])[0]
    upsert_chunks([{**row, "embedding": vector}])
    return str(row["source_file"])


def is_contribution_reply(
    *,
    author_is_bot: bool,
    referenced_author_is_bot: bool,
    referenced_text: str,
) -> bool:
    """A human reply to the bot's unknown message is a contribution. Other replies are not."""
    if author_is_bot or not referenced_author_is_bot:
        return False
    return UNKNOWN_INVITE in referenced_text


def normalize_question(text: str) -> str:
    """Case, whitespace, and punctuation do not change which question was saved."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


# Question-to-question cosine. "tickets assigned to" questions share most words,
# so a chunk-score floor of 0.30 treats them as the same. 0.85 keeps a headcount
# paraphrase and rejects a different person's ticket question.
QUESTION_MATCH_SIMILARITY = 0.85


def _is_contribution(result: object) -> bool:
    chunk = getattr(result, "chunk", None)
    if chunk is None:
        return False
    metadata = chunk.metadata or {}
    return metadata.get("source_type") == "user_contributed" or str(chunk.source_file).startswith(
        _SOURCE_PREFIX
    )


def _stored_question(result: object) -> str:
    chunk = result.chunk
    metadata = chunk.metadata or {}
    stored = str(metadata.get("question") or "")
    if stored.strip():
        return stored
    for line in chunk.chunk_text.splitlines():
        if line.lower().startswith("question:"):
            return line.split(":", 1)[1]
    return ""


def _when(result: object) -> str:
    metadata = result.chunk.metadata or {}
    return str(metadata.get("contributed_at") or metadata.get("date") or "")


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def question_similarity(asked: str, saved: str) -> float:
    """Cosine of the two question strings. A missing embedder does not match."""
    try:
        vectors = embed_texts([asked, saved])
    except (RuntimeError, ValueError):
        return 0.0
    if len(vectors) != 2:
        return 0.0
    return _cosine(vectors[0], vectors[1])


def _saved_body(chunk_text: str) -> tuple[str, str]:
    """The attribution line plus every line of the saved answer, and a citable quote."""
    lines = chunk_text.splitlines()
    quote_lines: list[str] = []
    capture = False
    for line in lines:
        if line.startswith("According to "):
            capture = True
        if capture:
            quote_lines.append(line)
    quote = "\n".join(quote_lines).strip()
    if not quote or quote not in chunk_text:
        quote = chunk_text.strip()
        return quote, quote
    body = "\n".join(line for line in quote_lines[1:] if line.strip()).strip()
    return body or quote, quote


# Second tier: same question signature (type, synonym classes, remaining words in
# order) plus a low cosine floor. The floor is only a backstop; the signature decides.
PARAPHRASE_MIN_SIMILARITY = 0.40

_COUNT_PHRASE = re.compile(
    r"\bhow (?:many|big|large)\b|\b(?:number|count|size) of\b|\bhead ?count\b|\b(?:team|staff) size\b"
)
_COUNT_IMPLIES_PEOPLE = re.compile(r"\bhead ?count\b|\b(?:team|staff) size\b")
_POLITE = re.compile(
    r"^(?:(?:hey|hi|hello|ok|okay|so|please|quick question)[,:!]?\s+)+"
    r"|\b(?:(?:can|could) you (?:please )?tell me|do you know|tell me|remind me)\b"
)
_FILLER = re.compile(
    r"\b(?:are|is) there\b|\bthere (?:are|is)\b|\bright now\b|\bat the moment\b|\bat present\b"
    r"|\bas of (?:now|today)\b|\bthese days\b|\bin total\b"
)
_QTYPE = {
    "who": "who", "whom": "who", "whose": "who", "what": "what", "which": "what",
    "when": "when", "where": "where", "why": "why", "how": "how",
}  # fmt: skip
_YES_NO = frozenset("is are am do does did can could will would should has have had any".split())
_COMPANY = frozenset("bmtech benchmark benchmarktech we our us company".split())
_PEOPLE = frozenset(
    "employee employees people person persons staff staffer staffers member members teammate "
    "teammates worker workers headcount ppl folks colleagues coworkers personnel".split()
)
_EMPLOYED = frozenset("work works working employ employs employed".split())
_STOP = frozenset(
    "a an the is are am do does have has got at in on of for to with by currently current now "
    "today presently total altogether exactly please".split()
)
_COUNT_WORDS = frozenset("what which how many big large".split())
_TYPO_TARGETS = {
    "company": ("bmtech", "benchmarktech"),
    "people": ("employee", "employees", "people", "members", "headcount", "teammates"),
}
_NOT_TYPOS = frozenset("employer employers employed".split())


def _typo_of(token: str, word: str) -> bool:
    """One edit or one adjacent swap, same first letter, five letters or more."""
    if len(token) < 5 or token[0] != word[0] or token in _NOT_TYPOS or abs(len(token) - len(word)) > 1:
        return False
    if len(token) == len(word):
        diff = [i for i, (a, b) in enumerate(zip(token, word, strict=True)) if a != b]
        if len(diff) == 1:
            return True
        return (
            len(diff) == 2
            and diff[1] == diff[0] + 1
            and token[diff[0]] == word[diff[1]]
            and token[diff[1]] == word[diff[0]]
        )
    short, long_ = (token, word) if len(token) < len(word) else (word, token)
    return any(long_[:i] + long_[i + 1 :] == short for i in range(len(long_)))


def _token_class(token: str, qtype: str) -> str:
    if token in _STOP or token in _COMPANY:
        return "drop"
    if token in _PEOPLE or (qtype == "count" and token == "team"):
        return "people"
    if token in _EMPLOYED:
        return "employed"
    for name, words in _TYPO_TARGETS.items():
        if any(_typo_of(token, word) for word in words):
            return "drop" if name == "company" else name
    return "term"


def question_signature(text: str) -> tuple[str, frozenset[str], tuple[str, ...]] | None:
    """Question type, synonym classes, and the remaining words in order.

    Unknown words stay terms, so an extra word (a scope, a tense, a different noun)
    always blocks a match. Company words are dropped: company scope is the default.
    """
    folded = text.casefold().replace("\u2019", "'").replace("\u2018", "'")
    folded = re.sub(r"\b(what|who|how|where|when|why)'s\b", r"\1 is", folded)
    folded = re.sub(r"'s\b", "", folded)
    folded = re.sub(r"\bbm[\s-]tech\b|\bbench ?mark[\s-]?tech\b", "bmtech", folded)
    folded = _POLITE.sub(" ", folded)
    classes: set[str] = set()
    qtype = None
    if _COUNT_PHRASE.search(folded):
        qtype = "count"
        if _COUNT_IMPLIES_PEOPLE.search(folded):
            classes.add("people")
        folded = _COUNT_PHRASE.sub(" ", folded)
    folded = _FILLER.sub(" ", folded)
    tokens = re.findall(r"#?[a-z0-9]+(?:[-_][a-z0-9]+)*", folded)
    if qtype is None:
        for index, token in enumerate(tokens):
            if token in _QTYPE or token in _YES_NO:
                qtype = _QTYPE.get(token, "yes_no")
                del tokens[index]
                break
    if qtype is None:
        return None
    terms: list[str] = []
    employed: list[str] = []
    for token in tokens:
        if qtype == "count" and token in _COUNT_WORDS:
            continue
        kind = _token_class(token, qtype)
        if kind == "drop":
            continue
        if kind == "people":
            classes.add("people")
        elif kind == "employed":
            employed.append(token)
        else:
            plural = len(token) > 4 and token.endswith("s") and not token.endswith("ss")
            terms.append(token[:-1] if plural else token)
    if not (qtype == "count" and "people" in classes):
        terms.extend(employed)  # "work" is filler only in a headcount question
    return (qtype, frozenset(classes), tuple(terms))


def same_question(asked: str, saved: str) -> bool:
    left, right = question_signature(asked), question_signature(saved)
    return left is not None and left == right and bool(left[1] or left[2])


@dataclass(frozen=True, slots=True)
class ContributionMatch:
    response: object  # AnswerResponse
    tier: str  # "exact" | "paraphrase"
    chunk_id: int
    similarity: float | None  # None for a normalised-equal match


def contribution_match(
    question: str, results: object, *, allow_paraphrase: bool = False
) -> ContributionMatch | None:
    """Pick a saved answer whose question matches the one just asked. No model call.

    Tier 1 (``exact``): equal after normalization, or a question-to-question cosine of
    at least ``QUESTION_MATCH_SIMILARITY``. Tier 2 (``paraphrase``, only when
    ``allow_paraphrase``): equal ``question_signature`` and a cosine of at least
    ``PARAPHRASE_MIN_SIMILARITY``. Any tier-1 match wins. Chunk rank is not a match.
    The similarity calls are the same, in the same order, as tier 1 alone.
    """
    hits = [result for result in results or () if _is_contribution(result)]
    if not hits:
        return None
    asked = normalize_question(question)
    exact: list[tuple[object, float | None]] = []
    loose: list[tuple[object, float | None]] = []
    for hit in hits:
        saved = _stored_question(hit)
        if normalize_question(saved) == asked:
            exact.append((hit, None))
            continue
        similarity = question_similarity(question, saved)
        if similarity >= QUESTION_MATCH_SIMILARITY:
            exact.append((hit, similarity))
        elif (
            allow_paraphrase
            and similarity >= PARAPHRASE_MIN_SIMILARITY
            and same_question(question, saved)
        ):
            loose.append((hit, similarity))
    tier, matched = ("exact", exact) if exact else ("paraphrase", loose)
    if not matched:
        return None
    chosen, similarity = max(matched, key=lambda item: (_when(item[0]), item[0].chunk.id))
    response = _saved_response(chosen)
    if response is None:
        return None
    return ContributionMatch(response, tier, chosen.chunk.id, similarity)


def contribution_answer(question: str, results: object) -> object | None:
    """Use a saved answer only when its question matches the one just asked (tier 1 only)."""
    match = contribution_match(question, results)
    return None if match is None else match.response


def _saved_response(chosen: object) -> object | None:
    """The saved answer, attributed and quoted, as a response with one citation."""
    from .models import AnswerResponse, Citation

    metadata = chosen.chunk.metadata or {}
    name = str(metadata.get("author_name") or "a teammate")
    when = str(metadata.get("date") or _when(chosen))[:10]
    body, quote = _saved_body(chosen.chunk.chunk_text)
    digest = (chosen.chunk.content_sha256 or "").strip()
    if not digest or not quote or quote not in chosen.chunk.chunk_text:
        return None
    answer = f"According to {name} on {when}: {body}"
    return AnswerResponse(
        status="answered",
        answer=answer,
        confidence=0.8,
        citations=[
            Citation(
                source_file=chosen.chunk.source_file,
                chunk_id=chosen.chunk.id,
                content_sha256=digest,
                quote=quote,
            )
        ],
    )


def remove_contribution(source_file: str) -> None:
    """Delete one stored contribution by its ``user_contributed/`` source path."""
    if not source_file.startswith(_SOURCE_PREFIX):
        raise ValueError("source_file must start with user_contributed/.")
    delete_source(source_file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_file", help="user_contributed/<channel>/<message>.md")
    args = parser.parse_args()
    remove_contribution(args.source_file)
