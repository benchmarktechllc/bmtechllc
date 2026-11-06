"""Token-aware chunking that keeps source boundaries and provenance intact."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import timedelta
from functools import lru_cache

from ..config import (
    CHUNK_MAX_TOKENS,
    CHUNK_MIN_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
    EMBEDDING_ENCODING,
)
from .clean import RawUnit


@dataclass(frozen=True, slots=True)
class Chunk:
    """A text chunk plus every source unit used to construct it."""

    text: str
    units: tuple[RawUnit, ...]
    token_count: int
    split_unit: bool = False


@lru_cache(maxsize=1)
def _encoding():
    try:
        import tiktoken
        return tiktoken.get_encoding(EMBEDDING_ENCODING)
    except Exception:  # Allows deterministic local inspection when encoding data is not cached.
        return None


def token_count(text: str) -> int:
    """Count embedding-model tokens, with a conservative offline fallback."""
    encoding = _encoding()
    if encoding is None:
        return len(re.findall(r"\w+|[^\w\s]", text))
    # Knowledge-base pages quote tokenizer control tokens such as <|endoftext|>.
    # Count those as ordinary characters so a documented token cannot abort ingestion.
    return len(encoding.encode(text, disallowed_special=()))


def _unit_boundary(previous: RawUnit, current: RawUnit) -> bool:
    """Return whether two units must not be joined into the same chunk."""
    if current.extra.get("boundary"):
        return True
    for key in ("section", "channel", "meeting"):
        if previous.extra.get(key) != current.extra.get(key):
            return True
    if previous.timestamp and current.timestamp:
        return current.timestamp - previous.timestamp > timedelta(minutes=30)
    return False


def _split_text(text: str, maximum: int) -> list[str]:
    """Split an oversize unit at paragraph then sentence boundaries."""
    # A giant code fence is one technical artifact; preserving it is safer than
    # shredding an executable example merely to satisfy a token target.
    if "```" in text:
        return [text]
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    fragments: list[str] = []
    for paragraph in paragraphs or [text]:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        buffer: list[str] = []
        for sentence in sentences:
            candidate = " ".join([*buffer, sentence]).strip()
            if buffer and token_count(candidate) > maximum:
                fragments.append(" ".join(buffer))
                buffer = [sentence]
            elif not buffer and token_count(sentence) > maximum:
                # Long unpunctuated text: split by words as a final deterministic fallback.
                words = sentence.split()
                word_buffer: list[str] = []
                for word in words:
                    word_candidate = " ".join([*word_buffer, word])
                    if word_buffer and token_count(word_candidate) > maximum:
                        fragments.append(" ".join(word_buffer))
                        word_buffer = [word]
                    else:
                        word_buffer.append(word)
                if word_buffer:
                    buffer = word_buffer
            else:
                buffer.append(sentence)
        if buffer:
            fragments.append(" ".join(buffer))
    return fragments


def _expand_oversize(units: Iterable[RawUnit], maximum: int) -> list[RawUnit]:
    expanded: list[RawUnit] = []
    for unit in units:
        if token_count(unit.text) <= maximum:
            expanded.append(unit)
            continue
        for fragment in _split_text(unit.text, maximum):
            extra = {**unit.extra, "split_unit": True}
            expanded.append(replace(unit, text=fragment, extra=extra))
    return expanded


def chunk_units(
    units: Iterable[RawUnit],
    *,
    target_tokens: int = CHUNK_TARGET_TOKENS,
    max_tokens: int = CHUNK_MAX_TOKENS,
    min_tokens: int = CHUNK_MIN_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk normalized units without crossing source boundaries.

    Overlap is applied only within a source boundary; chunks at a new section,
    channel, meeting, or a 30-minute Discord gap never borrow prior context.
    """
    if not 0 < min_tokens <= target_tokens <= max_tokens:
        raise ValueError("chunk token limits must satisfy 0 < min <= target <= max")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be non-negative and below max_tokens")

    values = _expand_oversize(units, max_tokens)
    chunks: list[Chunk] = []
    pending: list[RawUnit] = []

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        text = "\n".join(unit.text for unit in pending)
        chunks.append(
            Chunk(
                text=text,
                units=tuple(pending),
                token_count=token_count(text),
                split_unit=any(unit.extra.get("split_unit", False) for unit in pending),
            )
        )
        pending = []

    for unit in values:
        if not unit.text:
            continue
        if pending and _unit_boundary(pending[-1], unit):
            flush()
        candidate = "\n".join([*(item.text for item in pending), unit.text])
        pending_tokens = token_count("\n".join(item.text for item in pending))
        if pending and (
            token_count(candidate) > max_tokens
            or (token_count(candidate) > target_tokens and pending_tokens >= min_tokens)
        ):
            retained: list[RawUnit] = []
            if pending_tokens >= min_tokens:
                for old in reversed(pending):
                    if token_count("\n".join(item.text for item in [old, *retained])) > overlap_tokens:
                        break
                    retained.insert(0, old)
            flush()
            pending = retained
        pending.append(unit)
    flush()

    # Small trailing pieces are merged forward/backward where that does not
    # violate a hard provenance boundary or the hard model-token cap.
    merged: list[Chunk] = []
    for chunk in chunks:
        if (
            merged
            and merged[-1].token_count < min_tokens
            and not _unit_boundary(merged[-1].units[-1], chunk.units[0])
        ):
            text = f"{merged[-1].text}\n{chunk.text}"
            count = token_count(text)
            if count <= max_tokens:
                previous = merged.pop()
                merged.append(
                    Chunk(
                        text=text,
                        units=previous.units + chunk.units,
                        token_count=count,
                        split_unit=previous.split_unit or chunk.split_unit,
                    )
                )
                continue
        merged.append(chunk)
    return merged
