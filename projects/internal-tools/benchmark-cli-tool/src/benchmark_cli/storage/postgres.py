"""PostgreSQL persistence for chunk ingestion and independent retrieval modes."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import (
    DATABASE_URL,
    DB_STATEMENT_TIMEOUT_MS,
    HNSW_EF_SEARCH,
    PARENT_CONTEXT_MAX_TOKENS,
)
from ..models import GoldenChunkReference

SQL_DIR = Path(__file__).with_name("sql")


@dataclass(frozen=True, slots=True)
class SearchChunk:
    """A persisted chunk returned by one retrieval mode."""

    id: int
    source_file: str
    chunk_index: int
    chunk_text: str
    metadata: dict[str, Any]
    score: float
    token_count: int | None = None
    content_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ParentContext:
    """A hit plus nearby chunks from its containing source section."""

    anchor: SearchChunk
    section: str | None
    chunks: tuple[SearchChunk, ...]
    token_count: int
    section_complete: bool


@dataclass(frozen=True, slots=True)
class DiscordSourceMessage:
    """A normalized Discord message retained to rebuild one channel's chunks."""

    channel_id: str
    message_id: str
    channel_name: str
    author_name: str
    message_text: str
    sent_at: datetime
    edited_at: datetime | None = None
    reply_to_message_id: str | None = None


@dataclass(frozen=True, slots=True)
class PortalSyncState:
    """Last indexed revision of one portal knowledge-base page."""

    source_file: str
    content_sha256: str
    title: str
    section: str
    updated_on: Date | None = None


@dataclass(frozen=True, slots=True)
class FathomSyncState:
    """Last indexed revision of one Fathom meeting recording."""

    recording_id: str
    source_file: str
    content_sha256: str
    title: str
    fathom_url: str
    join_url: str
    attendees: tuple[str, ...]
    recorded_at: datetime | None
    revised_at: datetime | None


@dataclass(frozen=True, slots=True)
class TrelloSourceCard:
    """Last indexed revision of one Trello card."""

    board_id: str
    card_id: str
    source_file: str
    list_name: str
    card_name: str
    card_url: str
    labels: tuple[str, ...]
    due_at: datetime | None
    content_sha256: str
    last_activity_at: datetime | None


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """Optional provenance constraints shared by keyword and vector retrieval."""

    speaker: str | None = None
    source: str | None = None
    date: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (("speaker", self.speaker), ("source", self.source)):
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must not be blank when supplied.")
        if self.date is not None:
            try:
                Date.fromisoformat(self.date)
            except ValueError as exc:
                raise ValueError("date must use ISO-8601 YYYY-MM-DD format.") from exc


def _dependencies() -> tuple[Any, Any, Any]:
    try:
        import psycopg
        from pgvector.psycopg import register_vector
        from psycopg.types.json import Json
    except ImportError as exc:
        raise RuntimeError("PostgreSQL retrieval requires psycopg[binary] and pgvector.") from exc
    return psycopg, register_vector, Json


@contextmanager
def connection() -> Iterator[Any]:
    """Yield a configured pgvector connection."""
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured.")
    psycopg, register_vector, _ = _dependencies()
    with psycopg.connect(DATABASE_URL) as conn:
        try:
            register_vector(conn)
        except psycopg.ProgrammingError as exc:
            if "vector type not found" not in str(exc):
                raise
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT set_config('statement_timeout', %s, false)",
                (str(DB_STATEMENT_TIMEOUT_MS),),
            )
            cursor.execute("SELECT set_config('hnsw.ef_search', %s, false)", (str(HNSW_EF_SEARCH),))
        yield conn


def apply_sql(filename: str) -> None:
    """Apply one checked-in, idempotent SQL migration."""
    sql_path = SQL_DIR / filename
    if not sql_path.is_file():
        raise ValueError(f"Unknown SQL migration: {filename}")
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(sql_path.read_text(encoding="utf-8"))
        conn.commit()


def initialize_database() -> None:
    """Create or upgrade the chunk table, vector support, and FTS field."""
    apply_sql("001_init.sql")
    apply_sql("002_fts.sql")
    apply_sql("004_discord_sync.sql")
    apply_sql("005_portal_trello_sync.sql")
    apply_sql("006_fathom_sync.sql")


def create_indexes() -> None:
    """Create vector, FTS, and supporting indexes after ingestion."""
    apply_sql("003_indexes.sql")


def existing_hashes(hashes: Sequence[str]) -> set[str]:
    if not hashes:
        return set()
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(
            "SELECT content_sha256 FROM document_chunks WHERE content_sha256 = ANY(%s)",
            (list(hashes),),
        )
        return {row[0].strip() for row in cursor.fetchall()}


def existing_embeddings(hashes: Sequence[str]) -> dict[str, list[float]]:
    """Return persisted vectors keyed by content hash for an atomic source refresh."""
    if not hashes:
        return {}
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(
            "SELECT content_sha256, embedding FROM document_chunks WHERE content_sha256 = ANY(%s)",
            (list(hashes),),
        )
        return {
            content_sha256.strip(): embedding.to_list()
            for content_sha256, embedding in cursor.fetchall()
        }


def resolve_chunk_references(
    references: Sequence[GoldenChunkReference],
) -> dict[tuple[str, str], int]:
    """Resolve durable golden-dataset references to this database's surrogate IDs.

    A reference deliberately includes both source filename and content hash. This
    prevents a re-ingestion from silently measuring against an equally worded
    chunk from a different source and produces a clear failure when reviewed
    source text has changed.
    """
    keys = {(reference.source_file, reference.content_sha256) for reference in references}
    if not keys:
        return {}
    hashes = [content_hash for _, content_hash in keys]
    statement = """
        SELECT id, source_file, content_sha256
        FROM document_chunks
        WHERE content_sha256 = ANY(%s)
    """
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(statement, (hashes,))
        rows = cursor.fetchall()

    resolved = {(str(source_file), content_sha256.strip()): int(chunk_id) for chunk_id, source_file, content_sha256 in rows}
    missing = keys.difference(resolved)
    if missing:
        descriptions = ", ".join(
            f"{source_file} ({content_hash[:12]}…)"
            for source_file, content_hash in sorted(missing)
        )
        raise ValueError(
            "Golden dataset references are absent from the current corpus: "
            f"{descriptions}. Re-review the changed source content."
        )
    return {key: resolved[key] for key in keys}


def delete_source(source_file: str) -> None:
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute("DELETE FROM document_chunks WHERE source_file = %s", (source_file,))
        conn.commit()


def upsert_chunks(rows: Sequence[dict[str, Any]]) -> int:
    """Insert new chunks; the generated FTS column is derived by PostgreSQL."""
    if not rows:
        return 0
    _, _, Json = _dependencies()
    statement = """
        INSERT INTO document_chunks
            (source_file, chunk_index, chunk_text, content_sha256, token_count, embedding, metadata)
        VALUES (%(source_file)s, %(chunk_index)s, %(chunk_text)s, %(content_sha256)s,
                %(token_count)s, %(embedding)s, %(metadata)s)
        ON CONFLICT (content_sha256) DO NOTHING
    """
    prepared = [{**row, "metadata": Json(row["metadata"])} for row in rows]
    with connection() as conn, conn.cursor() as cursor:
        cursor.executemany(statement, prepared)
        inserted = cursor.rowcount
        conn.commit()
    return inserted


def replace_source_chunks(source_file: str, rows: Sequence[dict[str, Any]]) -> int:
    """Atomically replace one source after its embeddings have already been prepared.

    Generating embeddings happens before this function is called. Therefore, a
    provider failure cannot leave the source missing from the active corpus.
    """
    _, _, Json = _dependencies()
    statement = """
        INSERT INTO document_chunks
            (source_file, chunk_index, chunk_text, content_sha256, token_count, embedding, metadata)
        VALUES (%(source_file)s, %(chunk_index)s, %(chunk_text)s, %(content_sha256)s,
                %(token_count)s, %(embedding)s, %(metadata)s)
        ON CONFLICT (content_sha256) DO NOTHING
    """
    prepared = [{**row, "metadata": Json(row["metadata"])} for row in rows]
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute("DELETE FROM document_chunks WHERE source_file = %s", (source_file,))
        if prepared:
            cursor.executemany(statement, prepared)
            inserted = cursor.rowcount
        else:
            inserted = 0
        conn.commit()
    return inserted


def discord_sync_cursor(channel_id: str) -> str | None:
    """Return the newest Discord message ID committed for one channel."""
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(
            "SELECT last_message_id FROM discord_sync_state WHERE channel_id = %s", (channel_id,)
        )
        row = cursor.fetchone()
    return str(row[0]) if row and row[0] is not None else None


def set_discord_sync_cursor(channel_id: str, channel_name: str, message_id: str | None) -> None:
    """Commit a channel cursor only after its current index has been written."""
    statement = """
        INSERT INTO discord_sync_state (channel_id, channel_name, last_message_id)
        VALUES (%s, %s, %s)
        ON CONFLICT (channel_id) DO UPDATE
        SET channel_name = EXCLUDED.channel_name,
            last_message_id = EXCLUDED.last_message_id,
            reindex_required = FALSE,
            last_synced_at = now()
    """
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(statement, (channel_id, channel_name, message_id))
        conn.commit()


def upsert_discord_source_messages(messages: Sequence[DiscordSourceMessage]) -> int:
    """Store only newly observed or changed normalized Discord messages."""
    if not messages:
        return 0
    statement = """
        INSERT INTO discord_source_messages
            (channel_id, message_id, channel_name, author_name, message_text, sent_at,
             edited_at, reply_to_message_id)
        VALUES (%(channel_id)s, %(message_id)s, %(channel_name)s, %(author_name)s,
                %(message_text)s, %(sent_at)s, %(edited_at)s, %(reply_to_message_id)s)
        ON CONFLICT (channel_id, message_id) DO UPDATE
        SET channel_name = EXCLUDED.channel_name,
            author_name = EXCLUDED.author_name,
            message_text = EXCLUDED.message_text,
            sent_at = EXCLUDED.sent_at,
            edited_at = EXCLUDED.edited_at,
            reply_to_message_id = EXCLUDED.reply_to_message_id,
            updated_at = now()
        WHERE (discord_source_messages.channel_name,
               discord_source_messages.author_name,
               discord_source_messages.message_text,
               discord_source_messages.sent_at,
               discord_source_messages.edited_at,
               discord_source_messages.reply_to_message_id)
              IS DISTINCT FROM
              (EXCLUDED.channel_name,
               EXCLUDED.author_name,
               EXCLUDED.message_text,
               EXCLUDED.sent_at,
               EXCLUDED.edited_at,
               EXCLUDED.reply_to_message_id)
        RETURNING message_id
    """
    with connection() as conn, conn.cursor() as cursor:
        changed = 0
        for message in messages:
            cursor.execute(
                statement,
                {
                    "channel_id": message.channel_id,
                    "message_id": message.message_id,
                    "channel_name": message.channel_name,
                    "author_name": message.author_name,
                    "message_text": message.message_text,
                    "sent_at": message.sent_at,
                    "edited_at": message.edited_at,
                    "reply_to_message_id": message.reply_to_message_id,
                },
            )
            changed += int(cursor.fetchone() is not None)
        if changed:
            first = messages[0]
            cursor.execute(
                """
                INSERT INTO discord_sync_state (channel_id, channel_name, reindex_required)
                VALUES (%s, %s, TRUE)
                ON CONFLICT (channel_id) DO UPDATE
                SET channel_name = EXCLUDED.channel_name,
                    reindex_required = TRUE
                """,
                (first.channel_id, first.channel_name),
            )
        conn.commit()
    return changed


def discord_reindex_required(channel_id: str) -> bool:
    """Return whether stored message changes still need an index refresh."""
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(
            "SELECT reindex_required FROM discord_sync_state WHERE channel_id = %s", (channel_id,)
        )
        row = cursor.fetchone()
    return bool(row and row[0])


def discord_source_messages(channel_id: str) -> list[DiscordSourceMessage]:
    """Return a channel's normalized source messages in chronological order."""
    statement = """
        SELECT channel_id, message_id, channel_name, author_name, message_text, sent_at,
               edited_at, reply_to_message_id
        FROM discord_source_messages
        WHERE channel_id = %s
        ORDER BY sent_at ASC, message_id ASC
    """
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(statement, (channel_id,))
        return [DiscordSourceMessage(*row) for row in cursor.fetchall()]


def portal_sync_states() -> dict[str, PortalSyncState]:
    """Return every portal page revision recorded by ``benchmark-sync-portal``."""
    statement = """
        SELECT source_file, content_sha256, title, section, updated_on
        FROM portal_sync_state
    """
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(statement)
        rows = cursor.fetchall()
    return {
        str(source_file): PortalSyncState(
            str(source_file),
            str(content_sha256),
            str(title),
            str(section),
            updated_on,
        )
        for source_file, content_sha256, title, section, updated_on in rows
    }


def upsert_portal_sync_state(state: PortalSyncState) -> None:
    """Record a portal page after its chunks have been replaced."""
    statement = """
        INSERT INTO portal_sync_state
            (source_file, content_sha256, title, section, updated_on)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (source_file) DO UPDATE
        SET content_sha256 = EXCLUDED.content_sha256,
            title = EXCLUDED.title,
            section = EXCLUDED.section,
            updated_on = EXCLUDED.updated_on,
            last_synced_at = now()
    """
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(
            statement,
            (state.source_file, state.content_sha256, state.title, state.section, state.updated_on),
        )
        conn.commit()


def delete_portal_sync_state(source_file: str) -> None:
    """Forget a portal page after its chunks have been removed."""
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute("DELETE FROM portal_sync_state WHERE source_file = %s", (source_file,))
        conn.commit()


def trello_source_cards(board_id: str) -> dict[str, TrelloSourceCard]:
    """Return indexed cards for one Trello board, keyed by card id."""
    statement = """
        SELECT board_id, card_id, source_file, list_name, card_name, card_url,
               labels, due_at, content_sha256, last_activity_at
        FROM trello_source_cards
        WHERE board_id = %s
    """
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(statement, (board_id,))
        rows = cursor.fetchall()
    cards: dict[str, TrelloSourceCard] = {}
    for row in rows:
        labels = row[6] if isinstance(row[6], list) else []
        card = TrelloSourceCard(
            board_id=str(row[0]),
            card_id=str(row[1]),
            source_file=str(row[2]),
            list_name=str(row[3]),
            card_name=str(row[4]),
            card_url=str(row[5]),
            labels=tuple(str(label) for label in labels),
            due_at=row[7],
            content_sha256=str(row[8]),
            last_activity_at=row[9],
        )
        cards[card.card_id] = card
    return cards


def upsert_trello_source_card(card: TrelloSourceCard) -> None:
    """Record a Trello card after its chunks have been replaced."""
    _, _, Json = _dependencies()
    statement = """
        INSERT INTO trello_source_cards
            (board_id, card_id, source_file, list_name, card_name, card_url, labels,
             due_at, content_sha256, last_activity_at)
        VALUES (%(board_id)s, %(card_id)s, %(source_file)s, %(list_name)s, %(card_name)s,
                %(card_url)s, %(labels)s, %(due_at)s, %(content_sha256)s, %(last_activity_at)s)
        ON CONFLICT (board_id, card_id) DO UPDATE
        SET source_file = EXCLUDED.source_file,
            list_name = EXCLUDED.list_name,
            card_name = EXCLUDED.card_name,
            card_url = EXCLUDED.card_url,
            labels = EXCLUDED.labels,
            due_at = EXCLUDED.due_at,
            content_sha256 = EXCLUDED.content_sha256,
            last_activity_at = EXCLUDED.last_activity_at,
            updated_at = now()
    """
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(
            statement,
            {
                "board_id": card.board_id,
                "card_id": card.card_id,
                "source_file": card.source_file,
                "list_name": card.list_name,
                "card_name": card.card_name,
                "card_url": card.card_url,
                "labels": Json(list(card.labels)),
                "due_at": card.due_at,
                "content_sha256": card.content_sha256,
                "last_activity_at": card.last_activity_at,
            },
        )
        conn.commit()


def delete_trello_source_card(board_id: str, card_id: str) -> None:
    """Forget a Trello card after its chunks have been removed."""
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(
            "DELETE FROM trello_source_cards WHERE board_id = %s AND card_id = %s",
            (board_id, card_id),
        )
        conn.commit()


def fathom_sync_states() -> dict[str, FathomSyncState]:
    """Return every Fathom recording revision recorded by ``benchmark-sync-fathom``."""
    statement = """
        SELECT recording_id, source_file, content_sha256, title, fathom_url, join_url,
               attendees, recorded_at, revised_at
        FROM fathom_sync_state
    """
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(statement)
        rows = cursor.fetchall()
    states: dict[str, FathomSyncState] = {}
    for row in rows:
        attendees = row[6] if isinstance(row[6], list) else []
        state = FathomSyncState(
            recording_id=str(row[0]),
            source_file=str(row[1]),
            content_sha256=str(row[2]),
            title=str(row[3]),
            fathom_url=str(row[4]),
            join_url=str(row[5] or ""),
            attendees=tuple(str(person) for person in attendees),
            recorded_at=row[7],
            revised_at=row[8],
        )
        states[state.recording_id] = state
    return states


def upsert_fathom_sync_state(state: FathomSyncState) -> None:
    """Record a Fathom meeting after its chunks have been replaced."""
    _, _, Json = _dependencies()
    statement = """
        INSERT INTO fathom_sync_state
            (recording_id, source_file, content_sha256, title, fathom_url, join_url,
             attendees, recorded_at, revised_at)
        VALUES (%(recording_id)s, %(source_file)s, %(content_sha256)s, %(title)s,
                %(fathom_url)s, %(join_url)s, %(attendees)s, %(recorded_at)s, %(revised_at)s)
        ON CONFLICT (recording_id) DO UPDATE
        SET source_file = EXCLUDED.source_file,
            content_sha256 = EXCLUDED.content_sha256,
            title = EXCLUDED.title,
            fathom_url = EXCLUDED.fathom_url,
            join_url = EXCLUDED.join_url,
            attendees = EXCLUDED.attendees,
            recorded_at = EXCLUDED.recorded_at,
            revised_at = EXCLUDED.revised_at,
            last_synced_at = now()
    """
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(
            statement,
            {
                "recording_id": state.recording_id,
                "source_file": state.source_file,
                "content_sha256": state.content_sha256,
                "title": state.title,
                "fathom_url": state.fathom_url,
                "join_url": state.join_url,
                "attendees": Json(list(state.attendees)),
                "recorded_at": state.recorded_at,
                "revised_at": state.revised_at,
            },
        )
        conn.commit()


def delete_fathom_sync_state(recording_id: str) -> None:
    """Forget a Fathom meeting after its chunks have been removed."""
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute("DELETE FROM fathom_sync_state WHERE recording_id = %s", (recording_id,))
        conn.commit()


def set_trello_sync_state(
    board_id: str,
    board_name: str,
    short_link: str,
    last_activity_at: datetime | None,
) -> None:
    """Record a successful board snapshot after card indexing finishes."""
    statement = """
        INSERT INTO trello_sync_state (board_id, board_name, short_link, last_activity_at)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (board_id) DO UPDATE
        SET board_name = EXCLUDED.board_name,
            short_link = EXCLUDED.short_link,
            last_activity_at = EXCLUDED.last_activity_at,
            last_synced_at = now()
    """
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(statement, (board_id, board_name, short_link, last_activity_at))
        conn.commit()


def _filters_where_clause(filters: SearchFilters) -> tuple[str, list[str]]:
    """Return a parameterized SQL WHERE fragment for persisted provenance."""
    clauses: list[str] = []
    parameters: list[str] = []
    if filters.speaker is not None:
        clauses.append("metadata -> 'speakers' ? %s")
        parameters.append(filters.speaker)
    if filters.source is not None:
        clauses.append("source_file = %s")
        parameters.append(filters.source)
    if filters.date is not None:
        clauses.append(
            """(
                metadata ->> 'date' = %s
                OR metadata ->> 'date' = LEFT(%s, 4)
                OR (
                    metadata ->> 'date_start' <= %s
                    AND metadata ->> 'date_end' >= %s
                )
            )"""
        )
        parameters.extend((filters.date, filters.date, filters.date, filters.date))
    return (f"\n          AND {' AND '.join(clauses)}" if clauses else "", parameters)


def vector_search(
    query_vector: Sequence[float],
    limit: int = 20,
    *,
    speaker: str | None = None,
    source: str | None = None,
    date: str | None = None,
) -> list[SearchChunk]:
    """Return vector-ranked candidates independently of keyword retrieval."""
    if limit < 1:
        return []
    filter_clause, filter_parameters = _filters_where_clause(SearchFilters(speaker, source, date))
    statement = """
        SELECT id, source_file, chunk_index, chunk_text, metadata,
               1 - (embedding <=> %s::vector) AS score,
               NULL::integer AS token_count, content_sha256
        FROM document_chunks
        WHERE TRUE
        """ + filter_clause + """
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(
            statement,
            (list(query_vector), *filter_parameters, list(query_vector), limit),
        )
        return [SearchChunk(*row) for row in cursor.fetchall()]


def fts_search(
    query: str,
    limit: int = 20,
    *,
    speaker: str | None = None,
    source: str | None = None,
    date: str | None = None,
) -> list[SearchChunk]:
    """Return PostgreSQL full-text-ranked candidates independently of vectors."""
    if not query.strip() or limit < 1:
        return []
    filter_clause, filter_parameters = _filters_where_clause(SearchFilters(speaker, source, date))
    statement = """
        WITH query AS (
            SELECT replace(
                websearch_to_tsquery('english', %s)::text,
                ' & ',
                ' | '
            )::tsquery AS tsq
        )
        SELECT id, source_file, chunk_index, chunk_text, metadata,
               ts_rank_cd(search_vector, query.tsq) AS score,
               NULL::integer AS token_count, content_sha256
        FROM document_chunks, query
        WHERE search_vector @@ query.tsq
        """ + filter_clause + """
        ORDER BY score DESC, id ASC
        LIMIT %s
    """

    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(statement, (query, *filter_parameters, limit))
        return [SearchChunk(*row) for row in cursor.fetchall()]


def select_context_window(
    chunks: Sequence[SearchChunk], anchor_id: int, max_tokens: int
) -> tuple[tuple[SearchChunk, ...], int, bool]:
    """Return a contiguous, token-bounded window centered on an anchor chunk."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1.")
    try:
        anchor_index = next(index for index, chunk in enumerate(chunks) if chunk.id == anchor_id)
    except StopIteration as exc:
        raise ValueError(f"Anchor chunk {anchor_id} is not present in the supplied section.") from exc

    included = {anchor_index}
    total = chunks[anchor_index].token_count
    if total is None or total < 1:
        raise ValueError("Context chunks must provide a positive token_count.")

    left = anchor_index - 1
    right = anchor_index + 1
    while left >= 0 or right < len(chunks):
        added = False
        for side in ("left", "right"):
            candidate_index = left if side == "left" else right
            if candidate_index < 0 or candidate_index >= len(chunks):
                continue
            candidate_tokens = chunks[candidate_index].token_count
            if candidate_tokens is None or candidate_tokens < 1:
                raise ValueError("Context chunks must provide a positive token_count.")
            if total + candidate_tokens > max_tokens:
                if side == "left":
                    left = -1
                else:
                    right = len(chunks)
                continue
            included.add(candidate_index)
            total += candidate_tokens
            added = True
            if side == "left":
                left -= 1
            else:
                right += 1
        if not added:
            break

    selected = tuple(chunk for index, chunk in enumerate(chunks) if index in included)
    return selected, total, len(selected) == len(chunks)


def read_doc(chunk_id: int, *, max_tokens: int = PARENT_CONTEXT_MAX_TOKENS) -> ParentContext | None:
    """Fetch the hit and a token-bounded window from its containing section.

    The containing section is defined by ``source_file`` plus the persisted
    ``metadata.section`` value. When the section exceeds ``max_tokens``, the
    returned chunks are the nearest siblings around the hit and
    ``section_complete`` is false.
    """
    if chunk_id < 1:
        raise ValueError("chunk_id must be at least 1.")
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1.")
    anchor_statement = """
        SELECT id, source_file, chunk_index, chunk_text, metadata, 0.0 AS score,
               token_count
        FROM document_chunks
        WHERE id = %s
    """
    section_statement = """
        SELECT id, source_file, chunk_index, chunk_text, metadata, 0.0 AS score,
               token_count
        FROM document_chunks
        WHERE source_file = %s
          AND (metadata ->> 'section') IS NOT DISTINCT FROM %s
        ORDER BY chunk_index ASC, id ASC
    """
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(anchor_statement, (chunk_id,))
        anchor_row = cursor.fetchone()
        if anchor_row is None:
            return None
        anchor = SearchChunk(*anchor_row)
        section = anchor.metadata.get("section")
        if section is not None:
            section = str(section)
        cursor.execute(section_statement, (anchor.source_file, section))
        section_rows = cursor.fetchall()

    section_chunks = tuple(SearchChunk(*row) for row in section_rows)
    selected, total, complete = select_context_window(section_chunks, anchor.id, max_tokens)
    return ParentContext(anchor, section, selected, total, complete)
