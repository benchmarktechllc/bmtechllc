"""Offline tests for Trello board ingestion."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError

import pytest
from pydantic import ValidationError

from benchmark_cli.ingestion.chunk import chunk_units
from benchmark_cli.ingestion.pipeline import embedding_prefix, metadata_for
from benchmark_cli.models import ChunkMetadata
from benchmark_cli.storage.postgres import TrelloSourceCard
from benchmark_cli.trello_sync import (
    TrelloRestClient,
    parse_board,
    sync_board,
    trello_source_file,
    units_for_card,
)

_ACTIVITY = datetime(2026, 11, 1, tzinfo=UTC)
_BOARD = {
    "id": "board-1",
    "name": "BMTech General Board",
    "shortLink": "tIvVZqio",
    "url": "https://trello.com/b/tIvVZqio/bmtech-general-board",
    "lists": [{"id": "list-1", "name": "Doing"}],
    "cards": [
        {
            "id": "card-1",
            "name": "Ship the portal sync",
            "desc": "Use password=hunter2 in the notes.",
            "due": "2026-12-01T15:00:00.000Z",
            "shortUrl": "https://trello.com/c/abc123",
            "idList": "list-1",
            "labels": [{"name": "Ops", "color": "green"}, {"name": "", "color": "red"}],
            "dateLastActivity": "2026-11-01T00:00:00.000Z",
            "checklists": [
                {
                    "name": "Steps",
                    "checkItems": [
                        {"name": "Write docs", "state": "incomplete"},
                        {"name": "Review", "state": "complete"},
                    ],
                }
            ],
        },
        {
            "id": "card-archived",
            "name": "Hidden",
            "desc": "Should not be indexed.",
            "shortUrl": "https://trello.com/c/hidden",
            "idList": "archived-list",
        },
    ],
}


def _response(payload: object) -> object:
    class _Body:
        def read(self) -> bytes:
            return json.dumps(payload).encode()

        def __enter__(self) -> _Body:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    return _Body()


@pytest.mark.unit
def test_board_parser_keeps_open_cards_and_redacts_secrets() -> None:
    board = parse_board(_BOARD)

    assert board.short_link == "tIvVZqio"
    assert [card.card_id for card in board.cards] == ["card-1"]
    card = board.cards[0]
    assert card.labels == ("Ops", "red")
    assert card.list_name == "Doing"
    assert "[REDACTED]" in card.description
    assert "hunter2" not in card.description
    units = units_for_card(board, card)
    text = "\n".join(unit.text for unit in units)
    assert "Checklist: Steps" in text
    assert "- [ ] Write docs" in text
    assert "- [x] Review" in text
    assert trello_source_file(board.short_link, card.card_id) == "trello/tIvVZqio/card-1.md"

    chunk = chunk_units(units, target_tokens=400, max_tokens=500, min_tokens=1)[0]
    metadata = metadata_for(chunk, Path("trello/tIvVZqio/card-1.md"), "trello")
    payload = metadata.as_metadata()
    assert payload["source_type"] == "trello"
    assert payload["board"] == "BMTech General Board"
    assert payload["column"] == "Doing"
    assert payload["list"] == "Doing"
    assert payload["card_id"] == "card-1"
    assert payload["labels"] == ["Ops", "red"]
    assert payload["due"] == "2026-12-01"
    assert payload["url"] == "https://trello.com/c/abc123"
    assert "Trello card" in embedding_prefix(metadata, "card-1.md")


@pytest.mark.unit
def test_comments_page_until_a_short_page_and_stay_oldest_first() -> None:
    class FakeClient(TrelloRestClient):
        def __init__(self) -> None:
            super().__init__("key", "token")
            self.calls: list[dict[str, str] | None] = []

        def _request_json(self, path: str, params: dict[str, str] | None = None) -> object:
            assert path == "/cards/card-1/actions"
            self.calls.append(params)
            if params and params.get("before") == "cursor":
                return [
                    {
                        "id": "older",
                        "date": "2026-10-01T00:00:00.000Z",
                        "data": {"text": "password=nope"},
                        "memberCreator": {"fullName": "Ada Lovelace"},
                    }
                ]
            page = [
                {
                    "id": f"c{index}",
                    "date": "2026-11-01T00:00:00.000Z",
                    "data": {"text": f"note {index}"},
                    "memberCreator": {"username": "ada"},
                }
                for index in range(1000)
            ]
            page[-1]["id"] = "cursor"
            return page

    comments = FakeClient().card_comments("card-1")

    assert comments[0].text == "[REDACTED]"
    assert comments[0].author == "Ada Lovelace"
    assert comments[1].text == "note 999"
    assert comments[-1].text == "note 0"
    assert len(comments) == 1001


@pytest.mark.unit
def test_api_errors_do_not_include_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(request: object, timeout: int = 30) -> object:
        del request, timeout
        raise HTTPError(
            "https://api.trello.com/1/boards/tIvVZqio?key=SECRETKEY&token=SECRETTOKEN",
            401,
            "denied",
            None,
            io.BytesIO(b"{}"),
        )

    monkeypatch.setattr("benchmark_cli.trello_sync.urlopen", explode)
    client = TrelloRestClient("SECRETKEY", "SECRETTOKEN")

    with pytest.raises(Exception) as exc_info:
        client.fetch_board("tIvVZqio")

    message = str(exc_info.value)
    assert "SECRETKEY" not in message
    assert "SECRETTOKEN" not in message
    assert exc_info.value.__cause__ is None


@pytest.mark.unit
def test_sync_indexes_changed_cards_and_removes_missing_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    board = parse_board(_BOARD)
    card = board.cards[0]
    removed = TrelloSourceCard(
        board_id=board.board_id,
        card_id="gone",
        source_file="trello/tIvVZqio/gone.md",
        list_name="Done",
        card_name="Finished",
        card_url="https://trello.com/c/gone",
        labels=(),
        due_at=None,
        content_sha256="old",
        last_activity_at=_ACTIVITY,
    )
    unchanged = TrelloSourceCard(
        board_id=board.board_id,
        card_id=card.card_id,
        source_file="trello/tIvVZqio/card-1.md",
        list_name=card.list_name,
        card_name=card.name,
        card_url=card.url,
        labels=card.labels,
        due_at=card.due,
        content_sha256="existing",
        last_activity_at=card.last_activity,
    )

    class FakeClient:
        def __init__(self) -> None:
            self.comment_calls: list[str] = []

        def fetch_board(self, board_id: str) -> object:
            assert board_id == "tIvVZqio"
            return board

        def card_comments(self, card_id: str) -> tuple[object, ...]:
            self.comment_calls.append(card_id)
            return ()

    calls: list[object] = []
    monkeypatch.setattr("benchmark_cli.trello_sync.initialize_database", lambda: calls.append("init"))
    monkeypatch.setattr(
        "benchmark_cli.trello_sync.trello_source_cards",
        lambda board_id: {removed.card_id: removed, unchanged.card_id: unchanged}
        if board_id == board.board_id
        else {},
    )
    monkeypatch.setattr(
        "benchmark_cli.trello_sync.ingest_units",
        lambda units, **kwargs: calls.append(kwargs) or {"chunks": 3},
    )
    monkeypatch.setattr("benchmark_cli.trello_sync.delete_source", lambda source: calls.append(("delete", source)))
    monkeypatch.setattr(
        "benchmark_cli.trello_sync.delete_trello_source_card",
        lambda board_id, card_id: calls.append(("forget", board_id, card_id)),
    )
    monkeypatch.setattr(
        "benchmark_cli.trello_sync.upsert_trello_source_card",
        lambda row: calls.append(("upsert", row.card_id)),
    )
    monkeypatch.setattr(
        "benchmark_cli.trello_sync.set_trello_sync_state",
        lambda *args: calls.append(("state", args)),
    )
    monkeypatch.setattr("benchmark_cli.trello_sync.create_indexes", lambda: calls.append("indexes"))

    # Activity matches the stored card, so this snapshot should not reindex it.
    # A second card with new activity is added below by replacing the board.
    changed = board.cards[0]
    fresh = type(board)(
        board_id=board.board_id,
        name=board.name,
        short_link=board.short_link,
        url=board.url,
        cards=(
            changed,
            type(changed)(
                card_id="card-2",
                name="Ask about the board",
                description="Comments welcome.",
                list_name="Doing",
                labels=("Question",),
                due=None,
                url="https://trello.com/c/def456",
                last_activity=datetime(2026, 11, 2, tzinfo=UTC),
            ),
        ),
    )

    class SnapshotClient(FakeClient):
        def fetch_board(self, board_id: str) -> object:
            assert board_id == "tIvVZqio"
            return fresh

    client = SnapshotClient()
    reports = sync_board(client, "tIvVZqio")
    actions = {report.card_id: report.action for report in reports}

    assert actions["card-1"] == "unchanged"
    assert actions["card-2"] == "indexed"
    assert actions["gone"] == "removed"
    assert client.comment_calls == ["card-2"]
    indexed = next(item for item in calls if isinstance(item, dict))
    assert indexed["source_type"] == "trello"
    assert indexed["replace_existing"] is True
    assert indexed["source_filename"] == "trello/tIvVZqio/card-2.md"
    assert ("delete", "trello/tIvVZqio/gone.md") in calls
    assert ("forget", "board-1", "gone") in calls
    assert calls[-1] == "indexes"


@pytest.mark.unit
def test_trello_metadata_requires_card_provenance() -> None:
    with pytest.raises(ValidationError):
        ChunkMetadata(
            source_type="trello",
            date="2026-11-01",
            ingested_at="2026-11-01T00:00:00+00:00",
            board="BMTech General Board",
        )


@pytest.mark.unit
def test_sync_schema_declares_portal_and_trello_tables() -> None:
    sql_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "benchmark_cli"
        / "storage"
        / "sql"
        / "005_portal_trello_sync.sql"
    )
    sql = sql_path.read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS portal_sync_state" in sql
    assert "CREATE TABLE IF NOT EXISTS trello_source_cards" in sql
    assert "CREATE TABLE IF NOT EXISTS trello_sync_state" in sql
