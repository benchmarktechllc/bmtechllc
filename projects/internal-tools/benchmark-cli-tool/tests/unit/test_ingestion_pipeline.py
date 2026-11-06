"""Offline tests for ingestion behavior shared by file and live connectors."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from benchmark_cli.ingestion import pipeline
from benchmark_cli.ingestion.clean import RawUnit
from benchmark_cli.ingestion.pipeline import content_hash, ingest_units


@pytest.mark.unit
def test_replacing_a_source_reuses_embeddings_for_unchanged_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "VM configuration update. " * 40
    unit = RawUnit(
        text=text,
        speaker="Ada",
        timestamp=datetime(2026, 10, 23, tzinfo=UTC),
        ordinal=0,
        extra={"channel": "engineering", "section": "engineering"},
    )
    captured: list[dict[str, object]] = []
    vector = [0.1, 0.2]

    monkeypatch.setattr(pipeline, "initialize_database", lambda: None)
    monkeypatch.setattr(pipeline, "existing_embeddings", lambda hashes: {content_hash(text): vector})
    monkeypatch.setattr(
        pipeline,
        "embed_texts",
        lambda _: pytest.fail("unchanged content should reuse its existing embedding"),
    )
    monkeypatch.setattr(
        pipeline,
        "replace_source_chunks",
        lambda _source, rows: captured.extend(rows) or len(rows),
    )

    report = ingest_units(
        [unit],
        source_filename="discord-channel-1.json",
        source_type="discord",
        dry_run=False,
        replace_existing=True,
    )

    assert report["skipped"] == 1
    assert report["inserted"] == 1
    assert captured[0]["embedding"] == vector
