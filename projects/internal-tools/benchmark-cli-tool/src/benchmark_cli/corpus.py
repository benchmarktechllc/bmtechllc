"""Predicates for chunks that must not answer ordinary company questions.

Synthetic canaries are local evaluation transcripts. Bot abstentions and the
knowledge-base notification template are not evidence about the company.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from .ingestion.clean import message_is_exact_bot_template

_CANARY_FILE = re.compile(
    r"(?:^|/)(?:sample_meeting_transcript|sample_discord_chat)(?:\.[^/]+)?$"
    r"|(?:^|/)[^/]*(?:canary|synthetic)[^/]*$",
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

# Kept for callers that still import the name. Retrieval does not exclude
# human messages that happen to use these words.
BOILERPLATE_SQL = "FALSE"


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
    """Return whether the whole text is an exact bot template."""
    return message_is_exact_bot_template(text)
