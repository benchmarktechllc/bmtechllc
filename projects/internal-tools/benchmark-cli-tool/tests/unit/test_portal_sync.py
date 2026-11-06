"""Offline tests for portal knowledge-base ingestion."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmark_cli.ingestion.chunk import chunk_units
from benchmark_cli.ingestion.pipeline import embedding_prefix, metadata_for
from benchmark_cli.models import ChunkMetadata
from benchmark_cli.portal_sync import (
    index_entry_source,
    load_index_allowlist,
    load_portal_pages,
    page_digest,
    select_pages,
    sync_portal,
    units_for_page,
)
from benchmark_cli.storage.postgres import PortalSyncState

_PAGE = """---
title: 'Security Guide: Prompt Injection'
description: How to keep retrieved text from becoming instructions in an answer.
tags:
  - rag
  - security
status: published
visibility: internal
owner: ada
updated: '2026-09-07'
reviewCycleMonths: 6
---

Intro paragraph with password=hunter2.

## Response steps

Check the quote against the stored chunk.

## Empty later
"""


def _write_page(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _portal_content() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "portal" / "content"
        if candidate.is_dir():
            return candidate
    pytest.skip("portal content is not in this checkout")


@pytest.mark.unit
def test_frontmatter_parser_keeps_published_pages_and_redacts_secrets(tmp_path: Path) -> None:
    _write_page(tmp_path, "guide/security.md", _PAGE)
    _write_page(
        tmp_path,
        "playbooks/draft.md",
        _PAGE.replace("status: published", "status: draft"),
    )
    _write_page(
        tmp_path,
        "reference/old.md",
        _PAGE.replace("updated: '2026-09-07'", "updated: 2024-01-01").replace(
            "status: published", "status: archived"
        ),
    )

    pages = {page.source_file: page for page in load_portal_pages(tmp_path)}
    security = pages["portal/guide/security.md"]

    assert security.title == "Security Guide: Prompt Injection"
    assert security.tags == ("rag", "security")
    assert security.updated == date(2026, 9, 7)
    assert security.url == "/guide/security/"
    selected = select_pages(pages.values(), today=date(2026, 11, 6), allowlist=None)
    assert [page.source_file for page in selected] == ["portal/guide/security.md"]

    units = units_for_page(security)
    assert any("[REDACTED]" in unit.text for unit in units)
    assert all("hunter2" not in unit.text for unit in units)
    headings = [unit.extra["section"] for unit in units]
    assert "Response steps" in headings
    chunk = chunk_units(units, target_tokens=200, max_tokens=400, min_tokens=1)[0]
    metadata = metadata_for(chunk, Path(security.source_file), "kb_page")
    assert metadata.source_type == "kb_page"
    assert metadata.title == security.title
    assert metadata.kb_section == "guide"
    assert metadata.date == "2026-09-07"
    assert "Knowledge base page" in embedding_prefix(metadata, "security.md")


@pytest.mark.unit
def test_documented_tokenizer_control_tokens_still_chunk(tmp_path: Path) -> None:
    page = _PAGE.replace(
        "Intro paragraph with password=hunter2.",
        "cl100k_base reserves <|endoftext|> for a control signal.",
    )
    _write_page(tmp_path, "reference/tokens.md", page)

    reports = sync_portal(tmp_path, today=date(2026, 11, 6), dry_run=True)

    assert reports[0].chunks >= 1
    assert "<|endoftext|>" in " ".join(unit.text for unit in units_for_page(load_portal_pages(tmp_path)[0]))


@pytest.mark.unit
def test_stale_pages_follow_the_portal_month_count(tmp_path: Path) -> None:
    _write_page(tmp_path, "guide/fresh.md", _PAGE)
    pages = load_portal_pages(tmp_path)
    # Updated 2026-09-07 with a 6-month cycle: February is still fresh, March is stale.
    assert select_pages(pages, today=date(2027, 2, 1), allowlist=None) == pages
    assert select_pages(pages, today=date(2027, 3, 1), allowlist=None) == []


@pytest.mark.unit
def test_index_json_allowlist_drops_stale_and_missing_entries(tmp_path: Path) -> None:
    _write_page(tmp_path, "guide/security.md", _PAGE)
    _write_page(tmp_path, "guide/index.md", _PAGE.replace("Security Guide: Prompt Injection", "Guide"))
    index = tmp_path / "index.json"
    index.write_text(
        """[
          {"url": "/bmtechllc/portal/guide/security/", "section": "guide", "status": "published", "isStale": false},
          {"url": "/bmtechllc/portal/guide/", "section": "guide", "status": "published", "isStale": true}
        ]""",
        encoding="utf-8",
    )

    allow = load_index_allowlist(index)
    assert allow == {"portal/guide/security.md"}
    assert index_entry_source({"section": "guide", "url": "/guide/"}) == "portal/guide/index.md"
    selected = select_pages(
        load_portal_pages(tmp_path), today=date(2026, 11, 6), allowlist=allow
    )
    assert [page.source_file for page in selected] == ["portal/guide/security.md"]


@pytest.mark.unit
def test_sync_replaces_changed_pages_and_removes_dropped_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_page(tmp_path, "guide/security.md", _PAGE)
    page = load_portal_pages(tmp_path)[0]
    kept = PortalSyncState(
        source_file=page.source_file,
        content_sha256=page_digest(page),
        title=page.title,
        section="guide",
        updated_on=page.updated,
    )
    stale = PortalSyncState(
        source_file="portal/playbooks/retired.md",
        content_sha256="abc",
        title="Retired",
        section="playbooks",
        updated_on=date(2020, 1, 1),
    )
    calls: list[object] = []
    monkeypatch.setattr("benchmark_cli.portal_sync.initialize_database", lambda: calls.append("init"))
    monkeypatch.setattr(
        "benchmark_cli.portal_sync.portal_sync_states",
        lambda: {kept.source_file: kept, stale.source_file: stale},
    )
    monkeypatch.setattr(
        "benchmark_cli.portal_sync.ingest_units",
        lambda units, **kwargs: calls.append(kwargs) or {"chunks": 2},
    )
    monkeypatch.setattr(
        "benchmark_cli.portal_sync.delete_source",
        lambda source: calls.append(("delete", source)),
    )
    monkeypatch.setattr(
        "benchmark_cli.portal_sync.delete_portal_sync_state",
        lambda source: calls.append(("forget", source)),
    )
    monkeypatch.setattr(
        "benchmark_cli.portal_sync.upsert_portal_sync_state",
        lambda state: calls.append(("upsert", state.source_file)),
    )
    monkeypatch.setattr("benchmark_cli.portal_sync.create_indexes", lambda: calls.append("indexes"))

    _write_page(
        tmp_path,
        "reference/new.md",
        _PAGE.replace("guide", "reference").replace("Security Guide: Prompt Injection", "New page"),
    )
    reports = sync_portal(tmp_path, today=date(2026, 11, 6))
    actions = {report.source_file: report.action for report in reports}

    assert actions[page.source_file] == "unchanged"
    assert actions["portal/reference/new.md"] == "indexed"
    assert actions[stale.source_file] == "removed"
    indexed = next(item for item in calls if isinstance(item, dict))
    assert indexed["source_type"] == "kb_page"
    assert indexed["replace_existing"] is True
    assert indexed["source_filename"] == "portal/reference/new.md"
    assert ("delete", stale.source_file) in calls
    assert calls[-1] == "indexes"


@pytest.mark.unit
def test_dry_run_does_not_touch_the_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_page(tmp_path, "guide/security.md", _PAGE)
    monkeypatch.setattr(
        "benchmark_cli.portal_sync.initialize_database",
        lambda: pytest.fail("dry-run must not open the database"),
    )

    reports = sync_portal(tmp_path, today=date(2026, 11, 6), dry_run=True)

    assert reports[0].action == "dry_run"
    assert reports[0].chunks >= 1


@pytest.mark.unit
def test_kb_page_metadata_requires_section_and_title() -> None:
    with pytest.raises(ValidationError):
        ChunkMetadata(source_type="kb_page", date="2026-09-07", ingested_at="2026-09-07T00:00:00+00:00")


@pytest.mark.unit
def test_checked_in_portal_corpus_skips_drafts_and_chunks_published_pages() -> None:
    content = _portal_content()
    pages = {page.source_file: page for page in load_portal_pages(content)}
    security = pages[
        "portal/guide/security-guide-defending-against-indirect-prompt-injection-in-rag-architectures.md"
    ]
    assert "rag" in security.tags
    assert security.title.startswith("Security Guide:")

    reports = sync_portal(content, today=date(2026, 11, 6), dry_run=True)
    sources = {report.source_file for report in reports}
    assert "portal/guide/index.md" in sources
    assert "portal/playbooks/implementing-a-project-with-claude-code.md" not in sources
    assert "portal/reference/ai-engineering-topics.md" not in sources
    assert all(report.chunks >= 1 for report in reports)

    later = {
        report.source_file
        for report in sync_portal(content, today=date(2027, 2, 1), dry_run=True)
    }
    assert "portal/guide/adding-a-page-from-the-portal.md" not in later
    assert "portal/guide/index.md" in later

