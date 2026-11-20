"""Email mentions of a referenced teammate, read from the private chunk store.

No contact data lives in this repository. The roster supplies name labels, the database
supplies the messages, and the model decides what is relevant.
"""

from __future__ import annotations

import logging
import re

from .roster import person_by_label
from .storage.postgres import SearchChunk, person_email_rows

logger = logging.getLogger(__name__)

ID_BASE = 8_000_000_000_000
MAX_CONTACTS = 6
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_WORD = re.compile(r"[A-Za-z]+")


def name_parts(canonical: str) -> list[str]:
    """Lowercase alphabetic name tokens (4+ letters) from the person's roster labels."""
    person = person_by_label(canonical)
    labels = person.labels() if person else (canonical,)
    parts: list[str] = []
    for label in labels:
        for token in _WORD.findall(label or ""):
            token = token.lower()
            if len(token) >= 4 and token not in parts:
                parts.append(token)
    return parts


def contact_chunks(canonical: str) -> list[SearchChunk]:
    """One synthetic chunk per email mention: the line and up to two lines before it."""
    parts = name_parts(canonical)
    if not parts:
        return []
    try:
        rows = person_email_rows(parts, limit=MAX_CONTACTS)
    except Exception:
        logger.warning("Contact lookup is unavailable.")
        return []
    found: list[SearchChunk] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        lines = row.chunk_text.split("\n")
        k = 0
        for index, line in enumerate(lines):
            for match in _EMAIL.finditer(line):
                email = match.group(0)
                local = email.split("@", 1)[0].lower()
                if not any(part in local for part in parts):
                    continue
                snippet = "\n".join(lines[max(0, index - 2) : index + 1])
                key = (email.lower(), snippet)
                if key in seen or k >= 10:
                    continue
                seen.add(key)
                metadata = dict(row.metadata or {})
                metadata["contact_for"] = canonical
                metadata["derived_from_chunk_id"] = row.id
                found.append(
                    SearchChunk(
                        id=ID_BASE + row.id * 10 + k,
                        source_file=row.source_file,
                        chunk_index=row.chunk_index,
                        chunk_text=snippet,
                        metadata=metadata,
                        score=0.0,
                        content_sha256=row.content_sha256,
                    )
                )
                k += 1
                if len(found) >= MAX_CONTACTS:
                    return found
    return found
