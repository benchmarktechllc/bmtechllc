"""Synchronize one Trello board into the retrieval corpus.

Pulls open lists and cards from the Trello REST API, including descriptions,
checklists, and comments. Each card becomes one source,
``trello/<shortLink>/<cardId>.md``, and is replaced as a unit, the same way a
Discord channel is replaced. Cards that leave the open board are deleted from
``document_chunks``. A card whose ``dateLastActivity`` is unchanged since the
last successful index skips the comment fetch and the embedding call.

The default board is the BMTech General Board, short link ``tIvVZqio``.
Override it with ``TRELLO_BOARD_ID`` or ``--board``. ``benchmark-sync-trello``
also needs ``TRELLO_API_KEY``, ``TRELLO_TOKEN``, ``DATABASE_URL``, and
``OPENAI_API_KEY``. The ``trello-ingestion`` workflow runs on the same
``discord-ingestion`` GitHub environment as Discord sync. Never commit secrets.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .config import TRELLO_API_BASE_URL, TRELLO_API_KEY, TRELLO_BOARD_ID, TRELLO_TOKEN
from .ingestion.clean import RawUnit, normalize_text
from .ingestion.pipeline import content_hash, ingest_units
from .names import render_members
from .storage.postgres import (
    TrelloSourceCard,
    create_indexes,
    delete_source,
    delete_trello_source_card,
    initialize_database,
    set_trello_sync_state,
    trello_source_cards,
    upsert_trello_source_card,
)

TRELLO_SYNC_VERSION = "1"
_SAFE_ID = re.compile(r"[^A-Za-z0-9_-]+")
_BOARD_PAGE = re.compile(
    r"https://trello\.com/b/(?P<id>[A-Za-z0-9]+)/(?P<slug>[A-Za-z0-9-]+)/?",
    re.IGNORECASE,
)


class TrelloApiError(RuntimeError):
    """A Trello API request could not be completed."""


@dataclass(frozen=True, slots=True)
class TrelloComment:
    comment_id: str
    author: str
    text: str
    created_at: datetime | None


@dataclass(frozen=True, slots=True)
class TrelloChecklistItem:
    name: str
    complete: bool


@dataclass(frozen=True, slots=True)
class TrelloChecklist:
    name: str
    items: tuple[TrelloChecklistItem, ...]


@dataclass(frozen=True, slots=True)
class TrelloCard:
    card_id: str
    name: str
    description: str
    list_name: str
    labels: tuple[str, ...]
    due: datetime | None
    url: str
    last_activity: datetime | None
    checklists: tuple[TrelloChecklist, ...] = ()
    comments: tuple[TrelloComment, ...] = ()
    members: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TrelloBoard:
    board_id: str
    name: str
    short_link: str
    url: str
    cards: tuple[TrelloCard, ...]


@dataclass(frozen=True, slots=True)
class CardSyncReport:
    board_id: str
    card_id: str
    card_name: str
    source_file: str
    action: str
    chunks: int


class TrelloRestClient:
    """REST client for the board snapshot and card comments used by the sync."""

    def __init__(
        self,
        api_key: str,
        token: str,
        *,
        api_base_url: str = TRELLO_API_BASE_URL,
    ) -> None:
        self._api_key = api_key
        self._token = token
        self._api_base_url = api_base_url.rstrip("/")

    def _request_json(self, path: str, params: dict[str, str] | None = None) -> Any:
        query = dict(params or {})
        query["key"] = self._api_key
        query["token"] = self._token
        request = Request(
            f"{self._api_base_url}{path}?{urlencode(query)}",
            headers={
                "Accept": "application/json",
                "User-Agent": "benchmark-cli-tool trello-ingestion",
            },
        )
        for attempt in range(3):
            try:
                with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed Trello API URL
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                retry_after = _retry_after(exc)
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    # Drop the cause: HTTPError includes the URL, and that URL carries the key and token.
                    raise TrelloApiError(
                        f"Trello API request failed with HTTP {exc.code}."
                    ) from None
                time.sleep(max(retry_after, 2**attempt))
            except URLError:
                if attempt == 2:
                    raise TrelloApiError("Trello API request could not be reached.") from None
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def fetch_board(self, board_id: str) -> TrelloBoard:
        """Return open lists and cards. Comments are loaded separately per changed card."""
        payload = self._request_json(
            f"/boards/{quote(board_id, safe='')}",
            {
                "fields": "id,name,shortLink,url,shortUrl",
                "lists": "open",
                "list_fields": "name",
                "cards": "open",
                "card_fields": "name,desc,due,url,shortUrl,idList,labels,dateLastActivity,idMembers",
                "card_checklists": "all",
                "members": "all",
                "member_fields": "fullName,username",
            },
        )
        if not isinstance(payload, dict):
            raise TrelloApiError("Trello returned an invalid board response.")
        return parse_board(payload)

    def card_comments(self, card_id: str) -> tuple[TrelloComment, ...]:
        """Return comments oldest-first. Pages are bounded so a short page ends the scan."""
        before: str | None = None
        raw: list[dict[str, Any]] = []
        encoded = quote(card_id, safe="")
        for _page in range(20):
            params = {
                "filter": "commentCard",
                "limit": "1000",
                "fields": "id,date,data",
                "memberCreator": "true",
                "memberCreator_fields": "fullName,username",
            }
            if before:
                params["before"] = before
            payload = self._request_json(f"/cards/{encoded}/actions", params)
            if not isinstance(payload, list):
                raise TrelloApiError(
                    f"Trello returned an invalid comment response for card {card_id}."
                )
            page = [item for item in payload if isinstance(item, dict)]
            raw.extend(page)
            if len(payload) < 1000 or not page:
                break
            next_before = str(page[-1].get("id") or "")
            if not next_before or next_before == before:
                break
            before = next_before
        else:
            raise TrelloApiError(f"Trello comment pagination did not finish for card {card_id}.")
        comments = [parsed for item in reversed(raw) if (parsed := parse_comment(item)) is not None]
        return tuple(comments)


def parse_board(payload: dict[str, Any]) -> TrelloBoard:
    """Normalize a board payload. Cards on lists that are not open are omitted."""
    lists = payload.get("lists")
    cards = payload.get("cards")
    if not isinstance(lists, list) or not isinstance(cards, list):
        raise TrelloApiError("Trello returned a board payload without lists and cards.")
    list_names = {
        str(item.get("id")): normalize_text(str(item.get("name") or "")) or "Untitled list"
        for item in lists
        if isinstance(item, dict) and item.get("id")
    }
    parsed: list[TrelloCard] = []
    for item in cards:
        if not isinstance(item, dict):
            continue
        card = _parse_card(item, list_names, _member_names(payload))
        if card is not None:
            parsed.append(card)
    board_id = str(payload.get("id") or "").strip()
    short_link = str(payload.get("shortLink") or board_id or "board")
    if not board_id:
        raise TrelloApiError("Trello returned a board payload without an id.")
    return TrelloBoard(
        board_id=board_id,
        name=normalize_text(str(payload.get("name") or "")) or short_link,
        short_link=short_link,
        url=_board_url(payload, short_link),
        cards=tuple(parsed),
    )


def _member_names(payload: dict[str, Any]) -> dict[str, str]:
    names: dict[str, str] = {}
    members = payload.get("members")
    if not isinstance(members, list):
        return names
    for member in members:
        if not isinstance(member, dict) or not member.get("id"):
            continue
        label = normalize_text(str(member.get("fullName") or member.get("username") or ""))
        if label:
            names[str(member["id"])] = label
    return names


def _board_url(payload: dict[str, Any], short_link: str) -> str:
    """Prefer a ``/b/<id>/<slug>`` URL from ``url``, then ``shortUrl``."""
    candidates = (
        str(payload.get("url") or "").strip(),
        str(payload.get("shortUrl") or "").strip(),
    )
    for candidate in candidates:
        if _BOARD_PAGE.search(candidate) and "/invite/" not in candidate.casefold() and "ATTI" not in candidate:
            return candidate
    return next((candidate for candidate in candidates if candidate), f"https://trello.com/b/{short_link}")


def board_page_url(url: str) -> str | None:
    """Canonical board link. Invite tokens and slug-less short URLs are omitted."""
    if "/invite/" in url.casefold() or "ATTI" in url:
        return None
    match = _BOARD_PAGE.search(url)
    if match is None:
        return None
    return f"https://trello.com/b/{match.group('id')}/{match.group('slug')}"


def parse_comment(item: dict[str, Any]) -> TrelloComment | None:
    """Normalize one comment action. Empty or secret-only comments are dropped."""
    data = item.get("data") if isinstance(item.get("data"), dict) else {}
    text = normalize_text(str(data.get("text") or ""))
    if not text:
        return None
    member = item.get("memberCreator") if isinstance(item.get("memberCreator"), dict) else {}
    author = str(member.get("fullName") or member.get("username") or "unknown").strip() or "unknown"
    return TrelloComment(
        comment_id=str(item.get("id") or ""),
        author=author,
        text=text,
        created_at=_parse_time(item.get("date")),
    )


def trello_source_file(short_link: str, card_id: str) -> str:
    board = _SAFE_ID.sub("", short_link) or "board"
    card = _SAFE_ID.sub("", card_id) or "card"
    return f"trello/{board}/{card}.md"


def units_for_card(board: TrelloBoard, card: TrelloCard) -> list[RawUnit]:
    """Build retrieval units for one card. Every unit carries the same card provenance."""
    stamp = card.last_activity or card.due or datetime(1970, 1, 1, tzinfo=UTC)
    extra = _card_extra(board, card)
    units: list[RawUnit] = []
    ordinal = 0
    lines = [
        f"Card: {card.name}",
        f"Board: {board.name}",
        f"List: {card.list_name}",
        f"URL: {card.url}",
    ]
    page = board_page_url(board.url)
    if page:
        lines.append(f"Board URL: {page}")
    if card.labels:
        lines.append("Labels: " + ", ".join(card.labels))
    if card.due is not None:
        lines.append(f"Due: {card.due.date().isoformat()}")
    if card.members:
        lines.append("Members: " + render_members(card.members))
    if card.last_activity is not None:
        lines.append(f"Last activity: {card.last_activity.date().isoformat()}")
    if card.checklists:
        total = sum(len(checklist.items) for checklist in card.checklists)
        complete = sum(
            1 for checklist in card.checklists for item in checklist.items if item.complete
        )
        lines.append(f"Checklists: {len(card.checklists)} ({complete}/{total} items complete)")
    for comment in card.comments[-3:]:
        lines.append(f"Recent comment by {comment.author}: {comment.text[:180]}")
    if card.description:
        lines.extend(("", card.description))
    units.append(RawUnit("\n".join(lines), None, stamp, ordinal, extra))
    ordinal += 1
    for checklist in card.checklists:
        item_lines = [f"Checklist: {checklist.name}"]
        item_lines.extend(
            f"- [{'x' if item.complete else ' '}] {item.name}" for item in checklist.items
        )
        units.append(RawUnit("\n".join(item_lines), None, stamp, ordinal, extra))
        ordinal += 1
    for comment in card.comments:
        when = comment.created_at.date().isoformat() if comment.created_at else "unknown date"
        text = f"Comment by {comment.author} on {when}:\n{comment.text}"
        units.append(RawUnit(text, comment.author, stamp, ordinal, extra))
        ordinal += 1
    return units


def card_digest(card: TrelloCard, units: Sequence[RawUnit]) -> str:
    lines = [
        TRELLO_SYNC_VERSION,
        card.list_name,
        card.url,
        card.due.isoformat() if card.due else "",
        ",".join(card.labels),
        *(unit.text for unit in units),
    ]
    return content_hash("\n".join(lines))


def sync_board(
    client: TrelloRestClient,
    board_id: str,
    *,
    batch_size: int = 128,
) -> list[CardSyncReport]:
    """Index changed cards on one board and delete cards that are no longer open."""
    initialize_database()
    board = client.fetch_board(board_id)
    stored = trello_source_cards(board.board_id)
    reports: list[CardSyncReport] = []
    seen: set[str] = set()
    for card in board.cards:
        seen.add(card.card_id)
        source = trello_source_file(board.short_link, card.card_id)
        previous = stored.get(card.card_id)
        if previous is not None and _same_moment(previous.last_activity_at, card.last_activity):
            reports.append(
                CardSyncReport(board.board_id, card.card_id, card.name, source, "unchanged", 0)
            )
            continue
        full = replace(card, comments=client.card_comments(card.card_id))
        units = units_for_card(board, full)
        digest = card_digest(full, units)
        if (
            previous is not None
            and previous.content_sha256 == digest
            and previous.source_file == source
        ):
            _remember_card(board, full, source, digest)
            reports.append(
                CardSyncReport(board.board_id, card.card_id, card.name, source, "unchanged", 0)
            )
            continue
        if previous is not None and previous.source_file != source:
            delete_source(previous.source_file)
        report = ingest_units(
            units,
            source_filename=source,
            source_type="trello",
            dry_run=False,
            replace_existing=True,
            batch_size=batch_size,
        )
        _remember_card(board, full, source, digest)
        reports.append(
            CardSyncReport(
                board.board_id, card.card_id, card.name, source, "indexed", int(report["chunks"])
            )
        )
    for card_id, previous in stored.items():
        if card_id in seen:
            continue
        delete_source(previous.source_file)
        delete_trello_source_card(board.board_id, card_id)
        reports.append(
            CardSyncReport(
                board.board_id, card_id, previous.card_name, previous.source_file, "removed", 0
            )
        )
    latest = max(
        (card.last_activity for card in board.cards if card.last_activity is not None),
        default=None,
    )
    set_trello_sync_state(board.board_id, board.name, board.short_link, latest)
    create_indexes()
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--board",
        help="Trello board id or short link. Defaults to TRELLO_BOARD_ID, or tIvVZqio.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Embedding inputs per request (1-128).",
    )
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 128:
        raise SystemExit("--batch-size must be between 1 and 128.")
    if not TRELLO_API_KEY or not TRELLO_TOKEN:
        raise SystemExit("TRELLO_API_KEY and TRELLO_TOKEN are required.")
    board_id = args.board.strip() if args.board and args.board.strip() else TRELLO_BOARD_ID
    reports = sync_board(
        TrelloRestClient(TRELLO_API_KEY, TRELLO_TOKEN),
        board_id,
        batch_size=args.batch_size,
    )
    print(json.dumps([asdict(report) for report in reports], indent=2))


def _remember_card(board: TrelloBoard, card: TrelloCard, source: str, digest: str) -> None:
    upsert_trello_source_card(
        TrelloSourceCard(
            board_id=board.board_id,
            card_id=card.card_id,
            source_file=source,
            list_name=card.list_name,
            card_name=card.name,
            card_url=card.url,
            labels=card.labels,
            due_at=card.due,
            content_sha256=digest,
            last_activity_at=card.last_activity,
        )
    )


def _card_extra(board: TrelloBoard, card: TrelloCard) -> dict[str, Any]:
    return {
        "section": card.list_name,
        "board": board.name,
        "column": card.list_name,
        "card_id": card.card_id,
        "title": card.name,
        "labels": list(card.labels),
        "due": card.due.date().isoformat() if card.due else None,
        "url": card.url,
        "boundary": False,
    }


def _parse_card(
    item: dict[str, Any],
    list_names: dict[str, str],
    members_by_id: dict[str, str] | None = None,
) -> TrelloCard | None:
    card_id = str(item.get("id") or "").strip()
    list_name = list_names.get(str(item.get("idList") or ""))
    url = str(item.get("shortUrl") or item.get("url") or "").strip()
    if not card_id or not list_name or not url:
        return None
    directory = members_by_id or {}
    member_ids = item.get("idMembers") if isinstance(item.get("idMembers"), list) else []
    members = tuple(
        directory[str(member_id)] for member_id in member_ids if str(member_id) in directory
    )
    return TrelloCard(
        card_id=card_id,
        name=normalize_text(str(item.get("name") or "")) or "Untitled card",
        description=normalize_text(str(item.get("desc") or "")),
        list_name=list_name,
        labels=_labels(item.get("labels")),
        due=_parse_time(item.get("due")),
        url=url,
        last_activity=_parse_time(item.get("dateLastActivity")),
        checklists=_checklists(item.get("checklists")),
        members=members,
    )


def _labels(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    names: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            names.append(item.strip())
        elif isinstance(item, dict):
            name = str(item.get("name") or item.get("color") or "").strip()
            if name:
                names.append(name)
    return tuple(dict.fromkeys(names))


def _checklists(value: Any) -> tuple[TrelloChecklist, ...]:
    if not isinstance(value, list):
        return ()
    checklists: list[TrelloChecklist] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        items: list[TrelloChecklistItem] = []
        raw_items = item.get("checkItems") if isinstance(item.get("checkItems"), list) else []
        for check in raw_items:
            if not isinstance(check, dict):
                continue
            name = normalize_text(str(check.get("name") or ""))
            if not name:
                continue
            items.append(TrelloChecklistItem(name, str(check.get("state") or "") == "complete"))
        if not items:
            continue
        title = normalize_text(str(item.get("name") or "")) or "Checklist"
        checklists.append(TrelloChecklist(title, tuple(items)))
    return tuple(checklists)


def _parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _same_moment(left: datetime | None, right: datetime | None) -> bool:
    if left is None or right is None:
        return False
    if left.tzinfo is None:
        left = left.replace(tzinfo=UTC)
    if right.tzinfo is None:
        right = right.replace(tzinfo=UTC)
    return left.astimezone(UTC) == right.astimezone(UTC)


def _retry_after(exc: HTTPError) -> float:
    if exc.headers is None:
        return 0.0
    raw = exc.headers.get("Retry-After")
    if not raw:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


if __name__ == "__main__":
    main()
