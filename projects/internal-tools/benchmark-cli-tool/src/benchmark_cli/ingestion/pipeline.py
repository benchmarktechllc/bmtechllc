"""CLI for idempotent local-source ingestion into the pgvector chunk store."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..models import ChunkMetadata
from ..paths import DOCUMENTS_DIR
from ..providers.openai import embed_texts
from ..storage.postgres import (
    create_indexes,
    existing_embeddings,
    existing_hashes,
    initialize_database,
    replace_source_chunks,
    upsert_chunks,
)
from .chunk import Chunk, chunk_units
from .clean import RawUnit, parse_by_type

DEFAULT_SOURCE_DIR = DOCUMENTS_DIR


def source_type_for(path: Path) -> str:
    lowered = path.name.casefold()
    if "discord" in lowered or path.parent.name.casefold() == "discord":
        return "discord"
    if "transcript" in lowered or path.parent.name.casefold() in {"transcripts", "meetings"}:
        return "transcript"
    return "policy_doc"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _date_range(units: Iterable[RawUnit]) -> tuple[str | None, str | None]:
    values = [unit.timestamp for unit in units if unit.timestamp]
    if not values:
        return None, None
    return min(values).date().isoformat(), max(values).date().isoformat()


def _source_date(path: Path) -> str:
    """Choose a stable source date for policy documents without timestamps."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        text = ""
    iso_date = re.search(r"\b20\d{2}-\d{2}-\d{2}\b", text)
    if iso_date:
        return iso_date.group(0)
    for pattern in ("%B %d, %Y", "%b %d, %Y"):
        match = re.search(r"\b[A-Z][a-z]+\s+\d{1,2},\s+20\d{2}\b", text)
        if match:
            try:
                return datetime.strptime(match.group(0), pattern).date().isoformat()
            except ValueError:
                pass
    year = re.search(r"\b20\d{2}\b", text)
    return year.group(0) if year else datetime.fromtimestamp(path.stat().st_mtime, UTC).date().isoformat()


def _speakers_in_order(units: Iterable[RawUnit]) -> list[str]:
    seen: set[str] = set()
    speakers: list[str] = []
    for unit in units:
        if unit.speaker and unit.speaker not in seen:
            speakers.append(unit.speaker)
            seen.add(unit.speaker)
    return speakers


def _timezone_name(units: Iterable[RawUnit]) -> str | None:
    """Return the first explicit UTC offset preserved by a source parser."""
    for unit in units:
        if unit.timestamp and unit.timestamp.tzinfo:
            name = unit.timestamp.tzname()
            if name:
                return name
    return None


def metadata_for(chunk: Chunk, path: Path, source_type: str) -> ChunkMetadata:
    first = chunk.units[0]
    start, end = _date_range(chunk.units)
    common: dict[str, Any] = {
        "source_type": source_type,
        "date": start or _source_date(path),
        "ingested_at": datetime.now(UTC).isoformat(),
        "message_count": len(chunk.units),
        "split_unit": chunk.split_unit,
        "embed_prefix": "",
    }
    if source_type == "discord":
        fallback_date = start or _source_date(path)
        common.update(
            channel=str(first.extra.get("channel", path.stem)),
            section=str(first.extra.get("section", first.extra.get("channel", path.stem))),
            timezone=_timezone_name(chunk.units),
            date_start=fallback_date,
            date_end=end or fallback_date,
            speakers=_speakers_in_order(chunk.units),
        )
    elif source_type == "transcript":
        common.update(
            meeting=str(first.extra.get("meeting", path.stem)),
            speakers=_speakers_in_order(chunk.units),
        )
        if first.extra.get("section"):
            common["section"] = str(first.extra["section"])
    else:
        common.update(section=str(first.extra.get("section", path.stem)))
    return ChunkMetadata.model_validate(common)


def embedding_input(chunk: Chunk, metadata: ChunkMetadata, source_filename: str) -> str:
    """Create the provenance header supplied to the embedding model."""
    return f"{embedding_prefix(metadata, source_filename)}\n\n{chunk.text}"


def embedding_prefix(metadata: ChunkMetadata, source_filename: str) -> str:
    """Return the provenance header used only at embedding time."""
    if metadata.source_type == "policy_doc":
        header = f"Policy document: {source_filename}; section: {metadata.section}."
    elif metadata.source_type == "discord":
        timezone = f"; timezone: {metadata.timezone}" if metadata.timezone else ""
        header = f"Discord channel: {metadata.channel}; dates: {metadata.date_start} to {metadata.date_end}{timezone}."
    else:
        header = f"Meeting transcript: {metadata.meeting}; date: {metadata.date or 'unknown'}; section: {metadata.section or 'general'}."
    return header


def discover_sources(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".txt", ".json", ".md"}
    )


def ingest_units(
    units: Iterable[RawUnit],
    *,
    source_filename: str,
    source_type: str,
    dry_run: bool,
    replace_existing: bool = False,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Ingest normalized units from either a local file or a live source connector."""
    path = Path(source_filename)
    chunks = chunk_units(units)
    prepared = [
        (chunk_index, chunk, metadata_for(chunk, path, source_type))
        for chunk_index, chunk in enumerate(chunks)
        if chunk.text.strip()
    ]
    prepared = [
        (
            chunk_index,
            chunk,
            metadata.model_copy(update={"embed_prefix": embedding_prefix(metadata, path.name)}),
        )
        for chunk_index, chunk, metadata in prepared
    ]
    hashes = {content_hash(chunk.text) for _, chunk, _ in prepared}
    result: dict[str, Any] = {
        "source": source_filename,
        "source_type": source_type,
        "chunks": len(prepared),
        "tokens": sum(chunk.token_count for _, chunk, _ in prepared),
        "inserted": 0,
        "skipped": 0,
        "dry_run": dry_run,
    }
    if dry_run:
        return result

    initialize_database()
    if replace_existing:
        vectors_by_hash = existing_embeddings(hashes)
        pending_by_hash = {
            content_hash(chunk.text): (chunk_index, chunk, metadata)
            for chunk_index, chunk, metadata in prepared
            if content_hash(chunk.text) not in vectors_by_hash
        }
        pending = list(pending_by_hash.values())
        result["skipped"] = len(prepared) - len(pending)
    else:
        known_hashes = existing_hashes(hashes)
        pending = [
            (chunk_index, chunk, metadata)
            for chunk_index, chunk, metadata in prepared
            if content_hash(chunk.text) not in known_hashes
        ]
        result["skipped"] = len(prepared) - len(pending)
        vectors_by_hash: dict[str, list[float]] = {}
        if not pending:
            return result

    inputs = [embedding_input(chunk, metadata, path.name) for _, chunk, metadata in pending]
    vectors: list[list[float]] = []
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start : start + batch_size]
        for attempt in range(3):
            try:
                vectors.extend(
                    embed_texts(batch)
                )
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
    vectors_by_hash.update(
        {content_hash(chunk.text): vector for (_, chunk, _), vector in zip(pending, vectors, strict=True)}
    )
    row_entries = prepared if replace_existing else pending
    rows = [
        {
            "source_file": source_filename,
            "chunk_index": chunk_index,
            "chunk_text": chunk.text,
            "content_sha256": content_hash(chunk.text),
            "token_count": chunk.token_count,
            "metadata": metadata.model_dump(mode="json"),
            "embedding": vectors_by_hash[content_hash(chunk.text)],
        }
        for chunk_index, chunk, metadata in row_entries
    ]
    result["inserted"] = (
        replace_source_chunks(source_filename, rows) if replace_existing else upsert_chunks(rows)
    )
    return result


def ingest_source(path: Path, *, dry_run: bool, force: bool, batch_size: int = 128) -> dict[str, Any]:
    """Ingest one local source using the common normalized-unit pipeline."""
    result = ingest_units(
        parse_by_type(path),
        source_filename=path.name,
        source_type=source_type_for(path),
        dry_run=dry_run,
        replace_existing=force,
        batch_size=batch_size,
    )
    result["source"] = str(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--only", help="Glob pattern selecting source files (for example '*discord*.json').")
    parser.add_argument("--batch-size", type=int, default=128, help="Embedding inputs per request (1-128).")
    parser.add_argument("--dry-run", action="store_true", help="Parse/chunk and estimate cost without API or database writes.")
    parser.add_argument("--force", action="store_true", help="Replace existing chunks for each supplied source.")
    parser.add_argument("--create-indexes", action="store_true", help="Create HNSW/metadata indexes after ingestion.")
    args = parser.parse_args()
    if not args.source_dir.is_dir():
        raise SystemExit(f"Source directory does not exist: {args.source_dir}")
    if not 1 <= args.batch_size <= 128:
        raise SystemExit("--batch-size must be between 1 and 128.")
    sources = discover_sources(args.source_dir)
    if args.only:
        sources = [path for path in sources if path.match(args.only)]
    reports = [
        ingest_source(path, dry_run=args.dry_run, force=args.force, batch_size=args.batch_size)
        for path in sources
    ]
    if args.create_indexes and not args.dry_run:
        create_indexes()
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
