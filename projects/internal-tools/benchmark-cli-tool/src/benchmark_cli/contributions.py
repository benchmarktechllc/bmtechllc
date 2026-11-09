"""Remember an answer a person supplies when the bot cannot find one.

Contributions are embedded into ``document_chunks`` as ``user_contributed``.
They are not written into the git repository.
"""

from __future__ import annotations

import argparse
import re
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
