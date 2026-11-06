"""Synchronize published portal knowledge-base pages into the retrieval corpus.

Reads ``portal/content/**/*.md``, the files behind the portal ``/index.json``
manifest. A page is indexed only when frontmatter ``status`` is ``published``
and the page is not stale. Staleness matches the portal: whole calendar months
since ``updated`` are at least ``reviewCycleMonths`` (default 6). Draft,
archived, and stale pages are skipped. A page that was indexed before and is
no longer eligible is removed from ``document_chunks``.

Each page is one source, ``portal/<section>/<slug>.md``, and is replaced as a
unit, the same way a Discord channel is replaced. ``--index-json`` further
requires the built manifest entry to be published and not ``isStale``.

The ``portal-ingestion`` GitHub Actions workflow runs this daily and when
``portal/content`` changes on ``main``. It uses the existing
``discord-ingestion`` environment for ``DATABASE_URL`` and ``OPENAI_API_KEY``.
Optional ``PORTAL_CONTENT_DIR``, ``PORTAL_INDEX_JSON``, and ``PORTAL_BASE_URL``
select the corpus and the URL stored on each chunk. Do not commit secrets.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

from .ingestion.clean import RawUnit, normalize_text
from .ingestion.pipeline import content_hash, ingest_units
from .storage.postgres import (
    PortalSyncState,
    create_indexes,
    delete_portal_sync_state,
    delete_source,
    initialize_database,
    portal_sync_states,
    upsert_portal_sync_state,
)

PORTAL_SYNC_VERSION = "1"
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SECTIONS = frozenset(
    {"guide", "onboarding", "methodology", "playbooks", "runbooks", "reference", "decisions"}
)


@dataclass(frozen=True, slots=True)
class PortalPage:
    """One knowledge-base page selected from frontmatter and, optionally, index.json."""

    source_file: str
    section: str
    title: str
    description: str
    body: str
    tags: tuple[str, ...]
    visibility: str
    status: str
    owner: str
    updated: date
    review_cycle_months: int
    url: str

    def stale(self, today: date) -> bool:
        return months_between(self.updated, today) >= self.review_cycle_months


@dataclass(frozen=True, slots=True)
class PortalSyncReport:
    source_file: str
    title: str
    action: str
    chunks: int


def months_between(updated: date, today: date) -> int:
    """Whole calendar months, matching ``monthsSince`` in the portal."""
    return max(0, (today.year - updated.year) * 12 + (today.month - updated.month))


def default_portal_content_dir() -> Path:
    """Find ``portal/content`` from the working directory or an editable checkout."""
    override = os.getenv("PORTAL_CONTENT_DIR")
    if override and override.strip():
        return Path(override.strip())
    starts = [Path.cwd(), *Path(__file__).resolve().parents]
    for start in starts:
        for parent in (start, *start.parents):
            candidate = parent / "portal" / "content"
            if candidate.is_dir():
                return candidate
    raise SystemExit("portal/content was not found. Set PORTAL_CONTENT_DIR or pass --content-dir.")


def page_digest(page: PortalPage) -> str:
    """Stable hash of the fields that change chunk text or metadata."""
    payload = "\n".join(
        (
            PORTAL_SYNC_VERSION,
            page.source_file,
            page.title,
            page.description,
            page.updated.isoformat(),
            page.owner,
            page.visibility,
            page.status,
            ",".join(page.tags),
            page.url,
            page.body,
        )
    )
    return content_hash(payload)


def load_portal_pages(content_dir: Path) -> list[PortalPage]:
    """Parse every markdown page that has frontmatter. Unusable files are skipped."""
    pages: list[PortalPage] = []
    for path in sorted(content_dir.rglob("*.md")):
        if not path.is_file():
            continue
        page = _page_from_path(path, content_dir)
        if page is not None:
            pages.append(page)
    return pages


def load_index_allowlist(path: Path) -> set[str]:
    """Return source-file keys the built ``/index.json`` still considers trustworthy."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries: list[Any]
    if isinstance(payload, dict):
        raw = payload.get("pages", payload.get("entries", []))
        entries = raw if isinstance(raw, list) else []
    elif isinstance(payload, list):
        entries = payload
    else:
        raise SystemExit(f"Portal index must be a JSON list: {path}")
    allowed: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("status", "")).strip().lower() != "published":
            continue
        if entry.get("isStale") is True:
            continue
        key = index_entry_source(entry)
        if key:
            allowed.add(key)
    return allowed


def index_entry_source(entry: Mapping[str, Any]) -> str | None:
    """Map one ``/index.json`` record onto ``portal/<section>/<slug>.md``."""
    section = str(entry.get("section") or "").strip()
    if not section:
        return None
    url = str(entry.get("url") or "")
    parts = [part for part in url.split("/") if part]
    if section not in parts:
        markdown = str(entry.get("markdownUrl") or "")
        parts = [part for part in markdown.split("/") if part]
    if section not in parts:
        return f"portal/{section}/index.md"
    rest = parts[parts.index(section) + 1 :]
    if rest and rest[-1].endswith(".md"):
        rest[-1] = rest[-1][:-3]
    if not rest:
        return f"portal/{section}/index.md"
    return f"portal/{section}/{'/'.join(rest)}.md"


def select_pages(
    pages: Sequence[PortalPage],
    *,
    today: date,
    allowlist: set[str] | None,
) -> list[PortalPage]:
    """Keep published, fresh pages. An index allow-list can only remove pages."""
    selected: list[PortalPage] = []
    for page in pages:
        if page.status != "published" or page.stale(today):
            continue
        if allowlist is not None and not _index_allows(page.source_file, allowlist):
            continue
        selected.append(page)
    return selected


def units_for_page(page: PortalPage) -> list[RawUnit]:
    """Turn one page into section-bounded units. Secrets are redacted first."""
    stamp = datetime.combine(page.updated, time.min, tzinfo=UTC)
    units: list[RawUnit] = []
    ordinal = 0
    intro = normalize_text(f"{page.title}\n\n{page.description}")
    if intro:
        units.append(
            RawUnit(intro, None, stamp, ordinal, _page_extra(page, page.title, boundary=True))
        )
        ordinal += 1
    for heading, block in _heading_blocks(page.body):
        section = heading or page.title
        content = f"{heading}\n\n{block}" if heading and block else (heading or block)
        first = True
        for paragraph in re.split(r"\n\s*\n", content):
            text = normalize_text(paragraph)
            if not text:
                continue
            units.append(
                RawUnit(text, None, stamp, ordinal, _page_extra(page, section, boundary=first))
            )
            ordinal += 1
            first = False
    return units


def sync_portal(
    content_dir: Path,
    *,
    index_json: Path | None = None,
    today: date | None = None,
    dry_run: bool = False,
    batch_size: int = 128,
) -> list[PortalSyncReport]:
    """Index eligible pages and drop ones that are no longer published."""
    if not content_dir.is_dir():
        raise SystemExit(f"Portal content directory does not exist: {content_dir}")
    current_day = today or datetime.now(UTC).date()
    allowlist = load_index_allowlist(index_json) if index_json is not None else None
    pages = select_pages(load_portal_pages(content_dir), today=current_day, allowlist=allowlist)
    if dry_run:
        return [
            PortalSyncReport(
                page.source_file, page.title, "dry_run", _dry_run_chunks(page, batch_size)
            )
            for page in pages
        ]

    initialize_database()
    stored = portal_sync_states()
    reports: list[PortalSyncReport] = []
    seen: set[str] = set()
    for page in pages:
        seen.add(page.source_file)
        digest = page_digest(page)
        previous = stored.get(page.source_file)
        if previous is not None and previous.content_sha256 == digest:
            reports.append(PortalSyncReport(page.source_file, page.title, "unchanged", 0))
            continue
        units = units_for_page(page)
        if not units:
            if previous is not None:
                delete_source(page.source_file)
                delete_portal_sync_state(page.source_file)
            reports.append(PortalSyncReport(page.source_file, page.title, "empty", 0))
            continue
        report = ingest_units(
            units,
            source_filename=page.source_file,
            source_type="kb_page",
            dry_run=False,
            replace_existing=True,
            batch_size=batch_size,
        )
        upsert_portal_sync_state(
            PortalSyncState(
                source_file=page.source_file,
                content_sha256=digest,
                title=page.title,
                section=page.section,
                updated_on=page.updated,
            )
        )
        reports.append(
            PortalSyncReport(page.source_file, page.title, "indexed", int(report["chunks"]))
        )
    for source_file, previous in stored.items():
        if source_file in seen:
            continue
        delete_source(source_file)
        delete_portal_sync_state(source_file)
        reports.append(PortalSyncReport(source_file, previous.title, "removed", 0))
    create_indexes()
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--content-dir",
        type=Path,
        help="Portal markdown root. Defaults to portal/content.",
    )
    parser.add_argument(
        "--index-json",
        type=Path,
        help="Optional built /index.json. Published, non-stale entries further restrict the sync.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Embedding inputs per request (1-128).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk without API or database writes.",
    )
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 128:
        raise SystemExit("--batch-size must be between 1 and 128.")
    content_dir = args.content_dir or default_portal_content_dir()
    index_json = args.index_json
    if index_json is None and os.getenv("PORTAL_INDEX_JSON", "").strip():
        index_json = Path(os.environ["PORTAL_INDEX_JSON"].strip())
    if index_json is not None and not index_json.is_file():
        raise SystemExit(f"Portal index does not exist: {index_json}")
    reports = sync_portal(
        content_dir,
        index_json=index_json,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    )
    print(json.dumps([asdict(report) for report in reports], indent=2))


def _dry_run_chunks(page: PortalPage, batch_size: int) -> int:
    report = ingest_units(
        units_for_page(page),
        source_filename=page.source_file,
        source_type="kb_page",
        dry_run=True,
        replace_existing=True,
        batch_size=batch_size,
    )
    return int(report["chunks"])


def _index_allows(source_file: str, allowlist: set[str]) -> bool:
    aliases = {source_file}
    if source_file.endswith("/index.md"):
        aliases.add(source_file[: -len("/index.md")] + ".md")
    return bool(aliases & allowlist)


def _page_from_path(path: Path, content_dir: Path) -> PortalPage | None:
    try:
        relative = path.relative_to(content_dir)
    except ValueError:
        return None
    if len(relative.parts) < 2:
        return None
    section = relative.parts[0]
    if section not in _SECTIONS and not section.isidentifier():
        return None
    raw, body = _split_frontmatter(path.read_text(encoding="utf-8-sig"))
    if not raw:
        return None
    meta = _parse_frontmatter(raw)
    title = _text(meta.get("title"))
    updated = _parse_date(meta.get("updated"))
    if not title or updated is None:
        return None
    status = (_text(meta.get("status")) or "draft").lower()
    cycle = meta.get("reviewCycleMonths", meta.get("review_cycle_months", 6))
    try:
        review_cycle = int(cycle)
    except (TypeError, ValueError):
        review_cycle = 6
    if review_cycle <= 0:
        review_cycle = 6
    tags = _string_list(meta.get("tags"))
    return PortalPage(
        source_file=f"portal/{relative.as_posix()}",
        section=section,
        title=title,
        description=_text(meta.get("description")) or "",
        body=body.strip(),
        tags=tuple(tags),
        visibility=(_text(meta.get("visibility")) or "internal").lower(),
        status=status,
        owner=_text(meta.get("owner")) or "",
        updated=updated,
        review_cycle_months=review_cycle,
        url=_page_url(section, relative),
    )


def _page_url(section: str, relative: Path) -> str:
    rest = list(relative.with_suffix("").parts[1:])
    if not rest or rest == ["index"]:
        path = f"/{section}/"
    else:
        if rest[-1] == "index":
            rest = rest[:-1]
        path = f"/{section}/{'/'.join(rest)}/"
    base = os.getenv("PORTAL_BASE_URL", "").strip().rstrip("/")
    return f"{base}{path}" if base else path


def _page_extra(page: PortalPage, section: str, *, boundary: bool) -> dict[str, Any]:
    return {
        "section": section,
        "kb_section": page.section,
        "title": page.title,
        "url": page.url,
        "owner": page.owner or None,
        "tags": list(page.tags),
        "visibility": page.visibility,
        "kb_status": page.status,
        "boundary": boundary,
    }


def _heading_blocks(body: str) -> list[tuple[str | None, str]]:
    blocks: list[tuple[str | None, list[str]]] = []
    heading: str | None = None
    lines: list[str] = []
    for line in body.splitlines():
        match = _HEADING.match(line.strip())
        if match:
            text = "\n".join(lines).strip()
            if text:
                blocks.append((heading, text))
            heading = match.group(2).strip()
            lines = []
            continue
        lines.append(line)
    text = "\n".join(lines).strip()
    if text or (heading and not blocks):
        blocks.append((heading, text))
    return [(title, block) for title, block in blocks if block or title]


def _split_frontmatter(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return "", text
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "\n".join(lines[1:index]), "\n".join(lines[index + 1 :])
    return "", text


def _parse_frontmatter(raw: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    lines = raw.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if (
            not line.strip()
            or line.lstrip().startswith("#")
            or ":" not in line
            or line[:1].isspace()
        ):
            index += 1
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value in {"", "|", ">", "|-", ">-", "|+", ">+"}:
            items: list[Any] = []
            prose: list[str] = []
            index += 1
            while index < len(lines) and (not lines[index].strip() or lines[index][:1].isspace()):
                stripped = lines[index].strip()
                if stripped.startswith("- "):
                    items.append(_parse_scalar(stripped[2:]))
                elif stripped:
                    prose.append(stripped)
                index += 1
            data[key] = items if items else " ".join(prose)
            continue
        data[key] = _parse_scalar(value)
        index += 1
    return data


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if text in {"", "~", "null", "Null", "NULL"}:
        return None
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        return [_parse_scalar(part) for part in _split_list(text[1:-1]) if part.strip()]
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    return text


def _split_list(inner: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for char in inner:
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            continue
        if char == ",":
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        return None
    text = str(value).strip().strip("'\"")
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Iterable):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


if __name__ == "__main__":
    main()
