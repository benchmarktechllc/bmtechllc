from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx

from article_aggregator.config import FeedConfig
from article_aggregator.ingestion.fetch import USER_AGENT, fetch_feeds


def feed(name: str) -> FeedConfig:
    return FeedConfig(name=name, url=f"https://{name}.example/feed")


def run(feeds: list[FeedConfig], handler: object, **kwargs: int) -> list:  # type: ignore[type-arg]
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return asyncio.run(fetch_feeds(feeds, transport=transport, **kwargs))  # type: ignore[arg-type]


def test_one_failure_does_not_stop_others_and_order_is_kept() -> None:
    seen_agents: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_agents.append(request.headers["User-Agent"])
        host = request.url.host
        if host == "down.example":
            return httpx.Response(503)
        if host == "timeout.example":
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, content=b"<rss/>")

    results = run([feed("ok"), feed("down"), feed("timeout"), feed("ok2")], handler)

    assert [r.feed.name for r in results] == ["ok", "down", "timeout", "ok2"]
    assert [r.ok for r in results] == [True, False, False, True]
    assert results[0].content == b"<rss/>"
    assert "503" in (results[1].error or "")
    assert "ReadTimeout" in (results[2].error or "")
    assert set(seen_agents) == {USER_AGENT}


def test_declared_size_over_cap_fails() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Length": "999"}, content=b"x" * 999)

    [result] = run([feed("big")], handler, max_bytes=100)
    assert not result.ok
    assert "FeedTooLargeError" in (result.error or "")


def test_streamed_size_over_cap_fails() -> None:
    async def body() -> AsyncIterator[bytes]:
        yield b"x" * 60
        yield b"x" * 60

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body())

    [result] = run([feed("big")], handler, max_bytes=100)
    assert not result.ok
    assert "exceeds 100" in (result.error or "")


def test_concurrency_is_capped() -> None:
    active = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return httpx.Response(200, content=b"ok")

    results = run([feed(f"f{i}") for i in range(12)], handler, concurrency=5)
    assert all(r.ok for r in results)
    assert peak == 5
