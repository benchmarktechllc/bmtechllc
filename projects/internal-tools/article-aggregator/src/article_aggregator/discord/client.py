"""Minimal Discord REST client: post one message with a bot token, no gateway."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import httpx

log = logging.getLogger(__name__)

API_BASE = "https://discord.com/api/v10"
TIMEOUT_SECONDS = 15.0
MAX_RETRY_AFTER_SECONDS = 30.0


class DiscordError(Exception):
    pass


class DiscordClient:
    def __init__(
        self,
        token: str,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        base_url: str = API_BASE,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            transport=transport,
            timeout=TIMEOUT_SECONDS,
            headers={"Authorization": f"Bot {token}"},
        )
        self._sleep = sleep
        # Seconds to wait before the next post, from the last response's rate-limit headers.
        self._pause = 0.0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> DiscordClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def post_message(self, channel_id: str, payload: dict[str, Any]) -> str:
        """Post and return the new message ID.

        Waits out an exhausted rate-limit bucket before posting, and retries once on HTTP 429.
        """
        path = f"/channels/{channel_id}/messages"
        if self._pause > 0:
            self._sleep(self._pause)
            self._pause = 0.0
        response = self._post(path, payload)
        if response.status_code == 429:
            wait = min(_retry_after(response), MAX_RETRY_AFTER_SECONDS)
            log.warning("Discord rate limited the post; retrying once in %.1fs", wait)
            self._sleep(wait)
            response = self._post(path, payload)
        if response.status_code >= 400:
            raise DiscordError(
                f"Discord API returned HTTP {response.status_code}: {response.text[:300]}"
            )
        self._pause = _bucket_pause(response)
        return str(response.json()["id"])

    def _post(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        try:
            return self._client.post(path, json=payload)
        except httpx.HTTPError as exc:
            raise DiscordError(f"Discord request failed: {type(exc).__name__}: {exc}") from exc


def _bucket_pause(response: httpx.Response) -> float:
    """Reset delay when this response used the bucket's last request, else 0."""
    if response.headers.get("X-RateLimit-Remaining") != "0":
        return 0.0
    try:
        return min(float(response.headers.get("X-RateLimit-Reset-After", "1")),
                   MAX_RETRY_AFTER_SECONDS)
    except ValueError:
        return 1.0


def _retry_after(response: httpx.Response) -> float:
    try:
        return float(response.json()["retry_after"])
    except (ValueError, KeyError, TypeError):
        pass
    try:
        return float(response.headers.get("Retry-After", "1"))
    except ValueError:
        return 1.0
