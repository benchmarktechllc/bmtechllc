"""Index the public BMTech repository into ``github_chunks``.

Code chunks never enter ``document_chunks``, so indexing is safe on any build:
older builds and every non-code question never read this table. Run it before
or after a deploy.

Files are read from a local checkout (``--root``); no token is needed for that.
Pull requests come from the GitHub API. A ``GITHUB_TOKEN`` or ``GH_TOKEN``
already in the environment is sent when present; without one the public API
is used unauthenticated (60 requests an hour). ``--pulls-json`` reads a saved
pull-request list instead, and ``--skip-pulls`` leaves indexed pulls unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .github_files import (
    chunk_text,
    header_words,
    kind_for,
    render_chunk_text,
    render_embed_text,
    should_index,
)
from .ingestion.chunk import token_count
from .ingestion.clean import redact_secrets
from .providers.openai import embed_texts
from .storage.postgres import (
    delete_github_chunks,
    delete_github_file,
    github_existing_embeddings,
    github_file_digests,
    github_indexed_paths,
    initialize_database,
    replace_github_chunks,
    set_github_commit,
    upsert_github_file,
)

logger = logging.getLogger(__name__)
DEFAULT_REPO = "benchmarktechllc/bmtechllc"
_MAX_CHUNK_TOKENS = 1800
_EMBED_BATCH = 64


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


def row_digest(embed_text: str, stored_text: str) -> str:
    """sha256(embed_text || chunk_text). A change to either text re-embeds."""
    return hashlib.sha256((embed_text + stored_text).encode("utf-8")).hexdigest()


def code_rows(repo: str, commit_sha: str, path: str, text: str) -> list[dict[str, Any]]:
    """One github_chunks row per piece, with the stored text and the embedded text."""
    kind = kind_for(path)
    rows: list[dict[str, Any]] = []
    for piece in chunk_text(text, path):
        stored = _fit(
            redact_secrets(
                render_chunk_text(
                    repo, commit_sha, path, piece.start_line, piece.end_line, piece.symbol, piece.text
                )
            )
        )
        embedded = _fit(redact_secrets(render_embed_text(path, kind, piece)))
        if not stored.strip() or not embedded.strip():
            continue
        count = token_count(stored)
        if count < 1 or count > 2000:
            continue
        lines = stored.splitlines()
        url = lines[1] if len(lines) > 1 and lines[1].startswith("https://github.com/") else ""
        rows.append(
            {
                "kind": kind,
                "chunk_index": len(rows),
                "symbol": piece.symbol,
                "header_words": header_words(path, piece.symbol, piece.names),
                "start_line": piece.start_line,
                "end_line": piece.end_line,
                "chunk_text": stored,
                "embed_text": embedded,
                "content_sha256": row_digest(embedded, stored),
                "token_count": count,
                "embedding": [],
                "metadata": {"url": url, "commit_sha": commit_sha, "title": piece.symbol or path},
            }
        )
    return rows


def pull_rows(repo: str, commit_sha: str, record: dict[str, Any]) -> list[dict[str, Any]]:
    """One chunk for a pull request title, body, and changed paths. No diff."""
    number = int(record["number"])
    path = f"pulls/{number}"
    url = f"https://github.com/{repo}/pull/{number}"
    title = str(record.get("title") or "")
    files = "\n".join(f"- {item}" for item in record.get("files") or []) or "- (no files listed)"
    merged = str(record.get("merged_at") or "")
    body = str(record.get("body") or "").strip()
    stored = _fit(
        redact_secrets(
            "\n".join(
                [
                    f"Pull request #{number}: {title}",
                    f"State: {record.get('state') or ''}",
                    f"Merged: {merged}",
                    url,
                    "Files:",
                    files,
                    "",
                    body,
                ]
            )
            + "\n"
        )
    )
    embedded = _fit(redact_secrets(f"Pull request #{number}: {title}\nFiles:\n{files}\n\n{body}\n"))
    count = token_count(stored)
    if count < 1 or count > 2000:
        return []
    return [
        {
            "kind": "pull_request",
            "chunk_index": 0,
            "symbol": "",
            "header_words": header_words(path, "", [title]),
            "start_line": 1,
            "end_line": max(stored.count("\n"), 1),
            "chunk_text": stored,
            "embed_text": embedded,
            "content_sha256": row_digest(embedded, stored),
            "token_count": count,
            "embedding": [],
            "metadata": {
                "url": url,
                "merged_at": merged or None,
                "title": title or f"#{number}",
                "commit_sha": str(record.get("merge_sha") or commit_sha),
            },
        }
    ]


def sync_repository(
    root: Path,
    repo: str,
    commit_sha: str,
    *,
    pulls: list[dict[str, Any]] | None,
    full: bool = False,
) -> dict[str, int]:
    """Index changed files, and pull requests when a list is given. ``None`` leaves pulls alone."""
    initialize_database()
    recorded = github_file_digests(repo)
    present = github_indexed_paths(repo)
    # File state without rows (an older index wrote elsewhere) is not trusted.
    stored = {} if full else {path: digest for path, digest in recorded.items() if path in present}
    current_files = listed_files(root)
    for path in recorded:
        if path not in present and path not in current_files and not path.startswith("pulls/"):
            delete_github_file(repo, path)
    file_plan = plan_file_changes(
        {path: digest for path, digest in stored.items() if not path.startswith("pulls/")},
        {path: digest for path, (digest, _text) in current_files.items()},
    )
    indexed = 0
    for path in file_plan.changed:
        digest, text = current_files[path]
        rows = _with_embeddings(code_rows(repo, commit_sha, path, text))
        if rows:
            replace_github_chunks(repo, path, rows)
            upsert_github_file(repo, path, f"github/{repo}/{path}", digest)
            indexed += 1
            logger.info("indexed %s chunks=%s", path, len(rows))
        else:
            delete_github_chunks(repo, path)
            delete_github_file(repo, path)
    for path in file_plan.removed:
        delete_github_chunks(repo, path)
        delete_github_file(repo, path)
        logger.info("removed %s", path)
    pulls_indexed = 0
    pulls_removed = 0
    if pulls is not None:
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
            replace_github_chunks(repo, path, _with_embeddings(pull_rows_by_path[path]))
            upsert_github_file(repo, path, f"github/{repo}/{path}", pull_current[path])
            logger.info("indexed %s", path)
        for path in pull_plan.removed:
            delete_github_chunks(repo, path)
            delete_github_file(repo, path)
            logger.info("removed %s", path)
        pulls_indexed = len(pull_plan.changed)
        pulls_removed = len(pull_plan.removed)
    set_github_commit(repo, commit_sha)
    return {
        "files_indexed": indexed,
        "files_removed": len(file_plan.removed),
        "pulls_indexed": pulls_indexed,
        "pulls_removed": pulls_removed,
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
        paths = (
            [str(item.get("filename") or "") for item in files if isinstance(item, dict)]
            if isinstance(files, list)
            else []
        )
        records.append(
            {
                "number": number,
                "title": str(pull.get("title") or ""),
                "state": "merged" if pull.get("merged_at") else str(pull.get("state") or ""),
                "merged_at": str(pull.get("merged_at") or ""),
                "merge_sha": str(pull.get("merge_commit_sha") or ""),
                "body": str(pull.get("body") or ""),
                "files": [path for path in paths if path],
            }
        )
    return records


def _with_embeddings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reuse vectors from github_chunks by content hash; embed the rest from ``embed_text``."""
    if not rows:
        return []
    known = github_existing_embeddings([row["content_sha256"] for row in rows])
    pending = [row for row in rows if row["content_sha256"] not in known]
    for start in range(0, len(pending), _EMBED_BATCH):
        batch = pending[start : start + _EMBED_BATCH]
        vectors = embed_texts([row["embed_text"] for row in batch])
        for row, vector in zip(batch, vectors, strict=True):
            known[row["content_sha256"]] = vector
    for row in rows:
        row["embedding"] = known[row["content_sha256"]]
    return [{key: value for key, value in row.items() if key != "embed_text"} for row in rows]


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


def _pulls(args: argparse.Namespace) -> list[dict[str, Any]] | None:
    if args.pulls_json:
        payload = json.loads(Path(args.pulls_json).read_text(encoding="utf-8"))
        return list(payload)
    if args.skip_pulls:
        return None
    try:
        pulls = fetch_pull_requests(args.repo, _token())
    except Exception:
        logger.warning("Pull request list was not fetched. Indexed pulls are left unchanged.")
        return None
    if args.write_pulls_json:
        Path(args.write_pulls_json).write_text(json.dumps(pulls, indent=1) + "\n", encoding="utf-8")
    return pulls


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Local checkout of the public repository.")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--sha", default="", help="Commit SHA. Defaults to HEAD of --root.")
    parser.add_argument("--skip-pulls", action="store_true", help="Leave indexed pull requests unchanged.")
    parser.add_argument("--pulls-json", type=Path, help="Read the pull-request list from this file.")
    parser.add_argument("--write-pulls-json", type=Path, help="Save the fetched pull-request list here.")
    parser.add_argument("--full", action="store_true", help="Reindex every file, ignoring stored digests.")
    args = parser.parse_args()
    root = args.root.resolve()
    sha = args.sha.strip() or head_sha(root)
    summary = sync_repository(root, args.repo, sha, pulls=_pulls(args), full=args.full)
    logger.info(
        "github sync repo=%s sha=%s files_indexed=%s files_removed=%s pulls_indexed=%s pulls_removed=%s",
        args.repo,
        sha,
        summary["files_indexed"],
        summary["files_removed"],
        summary["pulls_indexed"],
        summary["pulls_removed"],
    )


if __name__ == "__main__":
    main()
