"""Live, read-only GitHub status for pull requests and issues named in a question.

Indexed pull request chunks hold title, state, files and body only. Reviews, checks and
authorship change after indexing, so a question that names PR/issue numbers gets one fresh
synthetic chunk per item. Every failure returns nothing and the caller carries on unchanged.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import urllib.error
from datetime import UTC, datetime
from typing import Any

from .github_sync import DEFAULT_REPO, _get_json, _token
from .storage.postgres import SearchChunk

logger = logging.getLogger(__name__)

# Citation requires chunk_id > 0; this base sits far above every database id sequence.
LIVE_ID_BASE = 9_000_000_000_000
LIVE_MARKER = "Live GitHub status fetched"
MAX_NUMBERS = 3
CACHE_SECONDS = 120.0
TIMEOUT_SECONDS = 5
API = "https://api.github.com"

_PHRASE = re.compile(
    r"(?<![\w<])(?:pull\s+requests?|prs?|issues?)\s*#?\s*(\d{1,5})\b", re.IGNORECASE
)
_HASH = re.compile(r"(?<![\w<&/])#(\d{1,5})\b")
_cache: dict[tuple[str, int], tuple[float, dict[str, Any] | None]] = {}


def referenced_numbers(text: str) -> list[int]:
    """PR or issue numbers named in ``text``, in order, de-duplicated, at most three."""
    found: list[tuple[int, int]] = []
    for pattern in (_PHRASE, _HASH):
        found.extend((match.start(), int(match.group(1))) for match in pattern.finditer(text or ""))
    numbers: list[int] = []
    for _, number in sorted(found):
        if number > 0 and number not in numbers:
            numbers.append(number)
    return numbers[:MAX_NUMBERS]


def _get(path: str) -> Any:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "benchmark-cli"}
    token = _token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return _get_json(f"{API}{path}", headers, timeout=TIMEOUT_SECONDS)


def _login(value: Any) -> str:
    return str((value or {}).get("login") or "") if isinstance(value, dict) else ""


def _pull_status(repo: str, number: int, pull: dict[str, Any]) -> dict[str, Any]:
    status: dict[str, Any] = {
        "kind": "pull request",
        "number": number,
        "title": str(pull.get("title") or ""),
        "url": str(pull.get("html_url") or f"https://github.com/{repo}/pull/{number}"),
        "author": _login(pull.get("user")),
        "state": str(pull.get("state") or ""),
        "merged": bool(pull.get("merged") or pull.get("merged_at")),
        "merged_at": pull.get("merged_at"),
        "merged_by": _login(pull.get("merged_by")),
        "created_at": pull.get("created_at"),
        "closed_at": pull.get("closed_at"),
        "updated_at": pull.get("updated_at"),
        "draft": bool(pull.get("draft")),
        "requested_reviewers": [_login(item) for item in pull.get("requested_reviewers") or []],
        "reviews": None,
        "checks": None,
    }
    try:
        latest: dict[str, dict[str, Any]] = {}
        for review in _get(f"/repos/{repo}/pulls/{number}/reviews?per_page=100") or []:
            user = _login(review.get("user"))
            if user and review.get("state") != "PENDING":
                latest[user] = {"user": user, "state": review.get("state"), "at": review.get("submitted_at")}
        status["reviews"] = list(latest.values())
    except Exception:
        logger.info("Live reviews for #%s were not fetched.", number)
    sha = str((pull.get("head") or {}).get("sha") or "")
    if sha:
        try:
            runs = (_get(f"/repos/{repo}/commits/{sha}/check-runs?per_page=100") or {}).get("check_runs") or []
            counts: dict[str, int] = {}
            failing: list[str] = []
            for run in runs:
                conclusion = str(run.get("conclusion") or run.get("status") or "unknown")
                counts[conclusion] = counts.get(conclusion, 0) + 1
                if conclusion in {"failure", "timed_out", "cancelled", "action_required"}:
                    failing.append(str(run.get("name") or ""))
            status["checks"] = {"counts": counts, "failing": failing}
        except Exception:
            logger.info("Live checks for #%s were not fetched.", number)
    return status


def _issue_status(repo: str, number: int, issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "issue",
        "number": number,
        "title": str(issue.get("title") or ""),
        "url": str(issue.get("html_url") or f"https://github.com/{repo}/issues/{number}"),
        "author": _login(issue.get("user")),
        "state": str(issue.get("state") or ""),
        "closed_at": issue.get("closed_at"),
        "created_at": issue.get("created_at"),
        "updated_at": issue.get("updated_at"),
        "labels": [str(item.get("name") or "") for item in issue.get("labels") or [] if isinstance(item, dict)],
        "assignees": [_login(item) for item in issue.get("assignees") or []],
    }


def fetch_status(number: int, repo: str = DEFAULT_REPO) -> dict[str, Any] | None:
    """Current status of a pull request, else an issue. None on any error. Cached ~2 minutes."""
    key = (repo, number)
    cached = _cache.get(key)
    if cached is not None and time.monotonic() - cached[0] < CACHE_SECONDS:
        return cached[1]
    try:
        try:
            result: dict[str, Any] | None = _pull_status(repo, number, _get(f"/repos/{repo}/pulls/{number}"))
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            issue = _get(f"/repos/{repo}/issues/{number}")
            result = _issue_status(repo, number, issue)
    except Exception:
        logger.info("Live GitHub status for #%s was not fetched.", number)
        return None
    _cache[key] = (time.monotonic(), result)
    return result


def _state_line(status: dict[str, Any]) -> str:
    if status["kind"] == "issue":
        if status["state"] == "closed" and status.get("closed_at"):
            return f"closed at {status['closed_at']}"
        return str(status["state"])
    if status["merged"]:
        by = f" by {status['merged_by']}" if status.get("merged_by") else ""
        return f"merged (merged at {status.get('merged_at')}{by})"
    return f"{status['state']}{' (draft)' if status['draft'] else ''}"


def _day(value: Any) -> str:
    return str(value or "")[:10] or "unknown date"


def _summary_line(status: dict[str, Any]) -> str:
    number = status["number"]
    if status["kind"] == "issue":
        day = f" on {_day(status.get('closed_at'))}" if status["state"] == "closed" else ""
        return f"Summary: Issue #{number} {status['state']}{day}; state {status['state']}"
    reviews = status.get("reviews") or []
    approved = [item["user"] for item in reviews if item.get("state") == "APPROVED"]
    changes = [item["user"] for item in reviews if item.get("state") == "CHANGES_REQUESTED"]
    if status["merged"]:
        by = f" by {status['merged_by']}" if status.get("merged_by") else ""
        head = f"PR #{number} merged on {_day(status.get('merged_at'))}{by}"
    elif status["state"] == "closed":
        head = f"PR #{number} closed without merging on {_day(status.get('closed_at') or status.get('updated_at'))}"
    else:
        head = f"PR #{number} {status['state']}{' draft' if status['draft'] else ''}"
    parts = [head]
    if approved:
        parts.append(f"approved by {', '.join(approved)}")
    if changes:
        parts.append(f"changes requested by {', '.join(changes)}")
    if not approved and not changes:
        parts.append("no approvals")
    parts.append(f"state {status['state']}")
    return "Summary: " + "; ".join(parts)


def format_status(status: dict[str, Any], fetched_at: str) -> str:
    """One fact per line; the core facts are adjacent so a quote of them is contiguous."""
    number = status["number"]
    kind = "Issue" if status["kind"] == "issue" else "Pull request"
    lines = [
        f"{kind} #{number}: {status['title']}",
        _summary_line(status),
        f"URL: {status['url']}",
        f"Author: {status['author'] or 'unknown'}",
        f"State: {_state_line(status)}",
    ]
    if status["kind"] == "issue":
        lines.append(f"Labels: {', '.join(status['labels']) or 'none'}")
        lines.append(f"Assignees: {', '.join(status['assignees']) or 'none'}")
    else:
        reviews = status.get("reviews")
        if reviews is None:
            lines.append("Reviews: unavailable")
        elif not reviews:
            lines.append("Reviews: none")
        else:
            lines.append(
                "Reviews: " + "; ".join(f"{item['user']}: {item['state']} at {item['at']}" for item in reviews)
            )
        if status.get("requested_reviewers"):
            lines.append(f"Requested reviewers: {', '.join(status['requested_reviewers'])}")
        checks = status.get("checks")
        if checks is None:
            lines.append("Checks: unavailable")
        elif not checks["counts"]:
            lines.append("Checks: none")
        else:
            lines.append("Checks: " + ", ".join(f"{count} {name}" for name, count in sorted(checks["counts"].items())))
            if checks["failing"]:
                lines.append(f"Failing checks: {', '.join(checks['failing'])}")
    lines.append(f"Created: {status.get('created_at')}")
    lines.append(f"Updated: {status.get('updated_at')}")
    lines.append(f"{LIVE_MARKER} {fetched_at}")
    return "\n".join(lines) + "\n"


def live_line_quote(quote: str, chunk_text: str) -> str | None:
    """Exact chunk text for a quote whose lines each sit inside some chunk line.

    Only for live status chunks. Matching lines are rebuilt from the chunk in chunk order; when
    they are not adjacent, the longest adjacent run is kept so the result is a chunk substring.
    """
    wanted = [line.strip() for line in quote.splitlines() if line.strip()]
    chunk_lines = chunk_text.split("\n")
    hits: set[int] = set()
    for line in wanted:
        index = next((i for i, text in enumerate(chunk_lines) if line in text), None)
        if index is None:
            return None
        hits.add(index)
    best: list[int] = []
    run: list[int] = []
    for index in sorted(hits):
        run = [*run, index] if run and index == run[-1] + 1 else [index]
        if len(run) > len(best):
            best = run
    return "\n".join(chunk_lines[i] for i in best) if best else None


def live_status_chunks(numbers: list[int], repo: str = DEFAULT_REPO) -> list[SearchChunk]:
    """One synthetic chunk per reachable PR or issue. Empty when nothing was fetched."""
    fetched_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    chunks: list[SearchChunk] = []
    for number in numbers[:MAX_NUMBERS]:
        status = fetch_status(number, repo)
        if status is None:
            continue
        text = format_status(status, fetched_at)
        path = f"{'pulls' if status['kind'] == 'pull request' else 'issues'}/{number}"
        chunks.append(
            SearchChunk(
                id=LIVE_ID_BASE + number,
                source_file=f"github/{path}",
                chunk_index=0,
                chunk_text=text,
                metadata={
                    "source_type": "github",
                    "kind": "pull_request" if status["kind"] == "pull request" else "issue",
                    "path": path,
                    "title": status["title"],
                    "url": status["url"],
                    "live": True,
                },
                score=1.0,
                token_count=None,
                content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
        )
    return chunks
