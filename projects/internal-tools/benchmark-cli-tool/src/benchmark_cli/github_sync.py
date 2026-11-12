"""Index the public BMTech repository into ``document_chunks``.

The first index belongs on the build that knows ``source_type`` github.
Older builds do not crash if these rows exist, but the rows can take
retrieval slots, so wait to index until this build is the one serving Discord.

No new secret. A ``GITHUB_TOKEN`` or ``GH_TOKEN`` already in the environment
is sent as a bearer token for pull-request reads. The public file tree is
read from ``--root``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .github_files import chunk_text, render_piece, should_index, source_file_for
from .ingestion.chunk import token_count
from .ingestion.clean import redact_secrets
from .ingestion.pipeline import content_hash
from .models import ChunkMetadata
from .providers.openai import embed_texts
from .storage.postgres import (
    create_indexes,
    delete_github_file,
    delete_source,
    existing_embeddings,
    github_file_digests,
    initialize_database,
    replace_source_chunks,
    set_github_commit,
    upsert_github_file,
)

logger = logging.getLogger(__name__)
DEFAULT_REPO = "benchmarktechllc/bmtechllc"
_MAX_CHUNK_TOKENS = 1800


@dataclass(frozen=True, slots=True)
class FilePlan:
    """Paths whose digest changed, and paths that left the tree."""

    changed: tuple[str, ...]
    removed: tuple[str, ...]


def plan_file_changes(previous: dict[str, str], current: dict[str, str]) -> FilePlan:
    """Reindex a path only when its digest changed. Delete paths that are gone."""
    removed = tuple(sorted(set(previous) - set(current)))
    changed = tuple(sorted(path for path, digest in current.items() if previous.get(path) != digest))
    return FilePlan(changed, removed)


def head_sha(root: Path) -> str:
    """Resolve HEAD from the git directory without requiring the git binary."""
    git_dir = root / ".git"
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref:"):
        return head
    ref = head.split(" ", 1)[1].strip()
    loose = git_dir / ref
    if loose.is_file():
        return loose.read_text(encoding="utf-8").strip()
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#") or line.startswith("^"):
                continue
            sha, _, name = line.partition(" ")
            if name.strip() == ref:
                return sha.strip()
    raise RuntimeError("Cannot resolve HEAD. Pass --sha.")


def listed_files(root: Path) -> dict[str, tuple[str, str]]:
    """Relative path to ``(content digest, utf-8 text)`` for files we index."""
    found: dict[str, tuple[str, str]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if not should_index(relative, size):
            continue
        try:
            data = path.read_bytes()
            text = data.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        found[relative] = (hashlib.sha256(data).hexdigest(), text)
    return found


def code_rows(repo: str, commit_sha: str, path: str, text: str) -> list[dict[str, Any]]:
    """One stored row per chunk. The row uses the existing document_chunks columns."""
    stamped = datetime.now(UTC).isoformat()
    day = stamped[:10]
    rows: list[dict[str, Any]] = []
    for piece in chunk_text(text, path):
        rendered = _fit(
            redact_secrets(
                render_piece(
                    repo,
                    commit_sha,
                    path,
                    piece.start_line,
                    piece.end_line,
                    piece.symbol,
                    piece.text,
                )
            )
        )
        if not rendered.strip():
            continue
        count = token_count(rendered)
        if count < 1 or count > 2000:
            continue
        url = rendered.splitlines()[1] if len(rendered.splitlines()) > 1 else ""
        metadata = ChunkMetadata(
            source_type="github",
            date=day,
            ingested_at=stamped,
            section=path,
            title=piece.symbol or path,
            repo=repo,
            path=path,
            commit_sha=commit_sha,
            url=url,
            start_line=piece.start_line,
            end_line=piece.end_line,
            symbol=piece.symbol or None,
            kind="code",
        )
        rows.append(_row(source_file_for(repo, path), len(rows), rendered, count, metadata))
    return rows


def pull_rows(repo: str, commit_sha: str, record: dict[str, Any]) -> list[dict[str, Any]]:
    """One chunk for a pull request title, body, and changed paths. No diff."""
    number = int(record["number"])
    path = f"pulls/{number}"
    url = f"https://github.com/{repo}/pull/{number}"
    files = "\n".join(f"- {item}" for item in record.get("files") or []) or "- (no files listed)"
    merged = str(record.get("merged_at") or "")
    text = _fit(
        redact_secrets(
            "\n".join(
                [
                    f"Pull request #{number}: {record.get('title') or ''}",
                    f"State: {record.get('state') or ''}",
                    f"Merged: {merged}",
                    url,
                    "Files:",
                    files,
                    "",
                    str(record.get("body") or "").strip(),
                ]
            )
            + "\n"
        )
    )
    count = token_count(text)
    if count < 1 or count > 2000:
        return []
    stamped = datetime.now(UTC).isoformat()
    metadata = ChunkMetadata(
        source_type="github",
        date=(merged or stamped)[:10],
        ingested_at=stamped,
        section=path,
        title=str(record.get("title") or f"#{number}"),
        repo=repo,
        path=path,
        commit_sha=str(record.get("merge_sha") or commit_sha),
        url=url,
        start_line=1,
        end_line=max(text.count("\n"), 1),
        kind="pull_request",
        merged_at=merged or None,
    )
    return [_row(source_file_for(repo, path), 0, text, count, metadata)]


def sync_repository(root: Path, repo: str, commit_sha: str, *, pulls: list[dict[str, Any]]) -> dict[str, int]:
    """Index changed files and the current pull-request list. Unchanged files are left alone."""
    initialize_database()
    stored = github_file_digests(repo)
    current_files = listed_files(root)
    file_plan = plan_file_changes(
        {path: digest for path, digest in stored.items() if not path.startswith("pulls/")},
        {path: digest for path, (digest, _text) in current_files.items()},
    )
    indexed = 0
    for path in file_plan.changed:
        digest, text = current_files[path]
        source = source_file_for(repo, path)
        rows = _with_embeddings(code_rows(repo, commit_sha, path, text))
        if rows:
            replace_source_chunks(source, rows)
            upsert_github_file(repo, path, source, digest)
            indexed += 1
            logger.info("indexed %s chunks=%s", path, len(rows))
        else:
            delete_source(source)
            delete_github_file(repo, path)
    for path in file_plan.removed:
        delete_source(source_file_for(repo, path))
        delete_github_file(repo, path)
        logger.info("removed %s", path)
    pull_current: dict[str, str] = {}
    pull_rows_by_path: dict[str, list[dict[str, Any]]] = {}
    for record in pulls:
        path = f"pulls/{int(record['number'])}"
        rows = pull_rows(repo, commit_sha, record)
        if not rows:
            continue
        pull_current[path] = rows[0]["content_sha256"]
        pull_rows_by_path[path] = rows
    pull_plan = plan_file_changes(
        {path: digest for path, digest in stored.items() if path.startswith("pulls/")},
        pull_current,
    )
    for path in pull_plan.changed:
        rows = _with_embeddings(pull_rows_by_path[path])
        source = source_file_for(repo, path)
        replace_source_chunks(source, rows)
        upsert_github_file(repo, path, source, pull_current[path])
        logger.info("indexed %s", path)
    for path in pull_plan.removed:
        delete_source(source_file_for(repo, path))
        delete_github_file(repo, path)
        logger.info("removed %s", path)
    set_github_commit(repo, commit_sha)
    create_indexes()
    return {
        "files_indexed": indexed,
        "files_removed": len(file_plan.removed),
        "pulls_indexed": len(pull_plan.changed),
        "pulls_removed": len(pull_plan.removed),
    }


def fetch_pull_requests(repo: str, token: str | None) -> list[dict[str, Any]]:
    """Titles, bodies, and changed paths. Diffs are not downloaded."""
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "bmtech-ask"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    pulls: list[dict[str, Any]] = []
    for page in range(1, 6):
        batch = _get_json(
            f"https://api.github.com/repos/{repo}/pulls?state=all&per_page=100&page={page}",
            headers,
        )
        if not isinstance(batch, list) or not batch:
            break
        pulls.extend(item for item in batch if isinstance(item, dict))
        if len(batch) < 100:
            break
    records: list[dict[str, Any]] = []
    for pull in pulls:
        number = int(pull["number"])
        files = _get_json(
            f"https://api.github.com/repos/{repo}/pulls/{number}/files?per_page=100",
            headers,
        )
        paths = [str(item.get("filename") or "") for item in files if isinstance(item, dict)] if isinstance(files, list) else []
        records.append(
            {
                "number": number,
                "title": str(pull.get("title") or ""),
                "state": "merged" if pull.get("merged_at") else str(pull.get("state") or ""),
                "merged_at": str(pull.get("merged_at") or ""),
                "merge_sha": str(pull.get("merge_commit_sha") or "" ),
                "body": str(pull.get("body") or ""),
                "files": [path for path in paths if path],
            }
        )
    return records


def _row(
    source_file: str, index: int, text: str, count: int, metadata: ChunkMetadata
) -> dict[str, Any]:
    return {
        "source_file": source_file,
        "chunk_index": index,
        "chunk_text": text,
        "content_sha256": content_hash(text),
        "token_count": count,
        "embedding": [],
        "metadata": metadata.as_metadata(),
    }


def _with_embeddings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    known = existing_embeddings([row["content_sha256"] for row in rows])
    pending = [row for row in rows if row["content_sha256"] not in known]
    if pending:
        vectors = embed_texts([row["chunk_text"] for row in pending])
        for row, vector in zip(pending, vectors, strict=True):
            known[row["content_sha256"]] = vector
    for row in rows:
        row["embedding"] = known[row["content_sha256"]]
    return rows


def _fit(text: str) -> str:
    if token_count(text) <= _MAX_CHUNK_TOKENS:
        return text
    lines = text.splitlines()
    while lines and token_count("\n".join(lines)) > _MAX_CHUNK_TOKENS:
        lines.pop()
    return "\n".join(lines).rstrip() + "\n"


def _get_json(url: str, headers: dict[str, str]) -> Any:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _token() -> str | None:
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Checkout of the public repository.")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--sha", default="", help="Commit SHA. Defaults to HEAD of --root.")
    parser.add_argument("--skip-pulls", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    sha = args.sha.strip() or head_sha(root)
    pulls: list[dict[str, Any]] = []
    if not args.skip_pulls:
        try:
            pulls = fetch_pull_requests(args.repo, _token())
        except Exception:
            logger.warning("Pull request list was not fetched. File index will still run.")
    summary = sync_repository(root, args.repo, sha, pulls=pulls)
    logger.info(
        "github sync repo=%s sha=%s files_indexed=%s files_removed=%s pulls_indexed=%s",
        args.repo,
        sha,
        summary["files_indexed"],
        summary["files_removed"],
        summary["pulls_indexed"],
    )


if __name__ == "__main__":
    main()
