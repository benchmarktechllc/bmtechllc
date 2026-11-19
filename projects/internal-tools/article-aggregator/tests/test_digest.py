from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from article_aggregator import cli
from article_aggregator.config import load_config
from article_aggregator.digest import AllFeedsFailedError, PartialPostError, run_digest
from article_aggregator.discord.client import DiscordError

from .conftest import NOW, fixture_bytes

FEEDS = {
    "news.example.com": "rss2.xml",
    "atom.example.org": "atom.xml",
}


def transport(overrides: dict[str, httpx.Response] | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if overrides and host in overrides:
            return overrides[host]
        if host in FEEDS:
            return httpx.Response(200, content=fixture_bytes(FEEDS[host]))
        raise AssertionError(f"unexpected request to {host}")  # disabled feeds are never fetched

    return httpx.MockTransport(handler)


def test_end_to_end_posts_ranked_recent_matches(config_dir: Path) -> None:
    posted: list[dict[str, Any]] = []

    def poster(payload: dict[str, Any]) -> str:
        posted.append(payload)
        return f"msg-{len(posted)}"

    result = run_digest(load_config(config_dir), now=NOW, poster=poster, transport=transport())

    assert result.message_ids == ["msg-1", "msg-2", "msg-3", "msg-4"]
    assert (result.feeds_ok, result.feeds_failed) == (2, 0)
    assert result.entries_seen == 7
    header, *articles = posted
    assert header["content"] == "**Digest 2026-11-18** · 3 articles"
    titles = [message["embeds"][0]["title"] for message in articles]
    # Two keywords first; then newest. Old, undated and non-matching items are excluded.
    assert titles == [
        "Supply chain attack hits package registry",
        "Hospital hit by ransomware attack",
        "Vendor patches Zero Day in VPN appliance",
    ]


def test_failure_partway_reports_how_many_were_posted(config_dir: Path) -> None:
    posted: list[dict[str, Any]] = []

    def poster(payload: dict[str, Any]) -> str:
        if len(posted) == 2:
            raise DiscordError("Discord API returned HTTP 500")
        posted.append(payload)
        return "m"

    with pytest.raises(PartialPostError, match="posted 2 of 4 digest messages") as excinfo:
        run_digest(load_config(config_dir), now=NOW, poster=poster, transport=transport())
    assert (excinfo.value.posted, excinfo.value.total) == (2, 4)


def test_dry_run_does_not_post(config_dir: Path) -> None:
    result = run_digest(load_config(config_dir), now=NOW, poster=None, transport=transport())
    assert len(result.payloads) == 4
    assert result.message_ids == []


def test_quiet_day_posts_nothing(config_dir: Path) -> None:
    calls: list[object] = []
    later = NOW.replace(year=2027)
    result = run_digest(load_config(config_dir), now=later, poster=calls.append,  # type: ignore[arg-type]
                        transport=transport())
    assert result.payloads == []
    assert calls == []


def test_failed_and_malformed_feeds_are_skipped(config_dir: Path) -> None:
    broken = {"atom.example.org": httpx.Response(200, content=fixture_bytes("malformed.xml"))}
    result = run_digest(load_config(config_dir), now=NOW, poster=None,
                        transport=transport(broken))
    assert (result.feeds_ok, result.feeds_failed) == (1, 1)
    assert result.matched == 2


def test_all_feeds_failing_is_an_error(config_dir: Path) -> None:
    down = {host: httpx.Response(500) for host in FEEDS}
    with pytest.raises(AllFeedsFailedError):
        run_digest(load_config(config_dir), now=NOW, poster=None, transport=transport(down))


def test_cli_requires_secrets_for_real_runs(config_dir: Path) -> None:
    code = cli.main(["digest", "--config-dir", str(config_dir)], env={})
    assert code == cli.EXIT_CONFIG


def test_cli_rejects_bad_config(tmp_path: Path) -> None:
    code = cli.main(["digest", "--dry-run", "--config-dir", str(tmp_path)], env={})
    assert code == cli.EXIT_CONFIG


def test_cli_target_picks_channel(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[str] = []

    def fake_run(config: object, *, now: object, poster: Any) -> Any:
        poster({"content": "x"})
        return run_digest_result()

    def fake_post(self: object, channel_id: str, payload: object) -> str:
        sent.append(channel_id)
        return "m"

    monkeypatch.setattr(cli, "run_digest", fake_run)
    monkeypatch.setattr(cli.DiscordClient, "post_message", fake_post)
    env = {
        "DISCORD_DIGEST_BOT_TOKEN": "t",
        "DISCORD_DIGEST_CHANNEL_ID": "prod",
        "DISCORD_DIGEST_TEST_CHANNEL_ID": "test",
    }
    assert cli.main(["digest", "--config-dir", str(config_dir)], env=env) == cli.EXIT_OK
    assert cli.main(["digest", "--target", "production", "--config-dir", str(config_dir)],
                    env=env) == cli.EXIT_OK
    assert sent == ["test", "prod"]


def run_digest_result() -> Any:
    from datetime import date

    from article_aggregator.digest import DigestResult

    return DigestResult(date(2026, 11, 18), 1, 0, 1, 1, 1, [{"content": "x"}], ["m"])
