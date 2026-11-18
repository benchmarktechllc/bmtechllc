"""Fetch feeds concurrently with a timeout, a size cap and per-feed failure isolation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

from article_aggregator import __version__
from article_aggregator.config import FeedConfig

log = logging.getLogger(__name__)

USER_AGENT = (
    f"BMTech-ArticleAggregator/{__version__} (+https://github.com/benchmarktechllc; RSS digest)"
)
TIMEOUT_SECONDS = 15.0
MAX_BYTES = 5 * 1024 * 1024
MAX_CONCURRENCY = 5


class FeedTooLargeError(Exception):
    pass


@dataclass(frozen=True)
class FetchResult:
    feed: FeedConfig
    content: bytes | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.content is not None


async def fetch_feeds(
    feeds: Sequence[FeedConfig],
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout: float = TIMEOUT_SECONDS,
    max_bytes: int = MAX_BYTES,
    concurrency: int = MAX_CONCURRENCY,
) -> list[FetchResult]:
    """Fetch every feed. One failure never stops the others; results keep input order."""
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(
        transport=transport,
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:

        async def one(feed: FeedConfig) -> FetchResult:
            async with semaphore:
                try:
                    content = await _get_capped(client, feed.url, max_bytes)
                except (httpx.HTTPError, FeedTooLargeError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    log.warning("feed %r failed: %s", feed.name, error)
                    return FetchResult(feed=feed, content=None, error=error)
                return FetchResult(feed=feed, content=content)

        return list(await asyncio.gather(*(one(feed) for feed in feeds)))


async def _get_capped(client: httpx.AsyncClient, url: str, max_bytes: int) -> bytes:
    async with client.stream("GET", url) as response:
        response.raise_for_status()
        declared = response.headers.get("Content-Length")
        if declared and declared.isdigit() and int(declared) > max_bytes:
            raise FeedTooLargeError(f"declared size {declared} bytes exceeds {max_bytes}")
        chunks: list[bytes] = []
        size = 0
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > max_bytes:
                raise FeedTooLargeError(f"body exceeds {max_bytes} bytes")
            chunks.append(chunk)
        return b"".join(chunks)
