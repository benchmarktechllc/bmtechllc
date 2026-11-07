"""Predicates for chunks that must not answer ordinary company questions.

Synthetic canaries are local evaluation transcripts. Bot abstentions and the
knowledge-base notification template are not evidence about the company.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

_CANARY_FILE = re.compile(
    r"(?:^|/)(?:sample_meeting_transcript|sample_discord_chat)(?:\.[^/]+)?$"
    r"|(?:^|/)[^/]*(?:canary|synthetic)[^/]*$",
    re.IGNORECASE,
)
_BOILERPLATE_TEXT = re.compile(
    r"no relevant information was found in the company document library"
    r"|there is no information available regarding"
    r"|retrieved evidence does not contain any specific link"
    r"|i couldn't answer that just now"
    r"|new documentation published",
    re.IGNORECASE,
)
_TRUE = frozenset({"true", "t", "1", "yes"})

SYNTHETIC_CORPUS_SQL = """
(
    lower(coalesce(metadata ->> 'corpus', '')) = 'canary'
    OR lower(coalesce(metadata ->> 'synthetic', '')) IN ('true', 't', '1', 'yes')
    OR source_file ~* '(^|/)(sample_meeting_transcript|sample_discord_chat)(\\.[^/]+)?$'
    OR source_file ~* '(^|/)[^/]*(canary|synthetic)[^/]*$'
    OR (
        coalesce(metadata ->> 'source_type', '') = 'transcript'
        AND coalesce(metadata ->> 'kind', '') <> 'meeting_transcript'
    )
)
"""

BOILERPLATE_SQL = """
(
    chunk_text ~* 'no relevant information was found in the company document library'
    OR chunk_text ~* 'there is no information available regarding'
    OR chunk_text ~* 'retrieved evidence does not contain any specific link'
    OR chunk_text ~* 'i couldn''t answer that just now'
    OR chunk_text ~* 'new documentation published'
)
"""


def is_synthetic_canary(source_file: str, metadata: Mapping[str, object] | None = None) -> bool:
    """Return whether a chunk is an evaluation canary rather than company evidence."""
    meta = metadata or {}
    if str(meta.get("corpus") or "").casefold() == "canary":
        return True
    if str(meta.get("synthetic") or "").casefold() in _TRUE or meta.get("synthetic") is True:
        return True
    if _CANARY_FILE.search(source_file.replace("\\", "/")):
        return True
    return (
        str(meta.get("source_type") or "") == "transcript"
        and str(meta.get("kind") or "") != "meeting_transcript"
    )


def is_boilerplate_text(text: str) -> bool:
    """Return whether text is an indexed bot abstention or notification template."""
    return _BOILERPLATE_TEXT.search(text) is not None
