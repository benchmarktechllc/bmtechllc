from __future__ import annotations

import httpx
import pytest

from article_aggregator.discord.client import DiscordClient, DiscordError

PAYLOAD = {"content": "hi", "embeds": [], "allowed_mentions": {"parse": []}}


def client_for(responses: list[httpx.Response], sleeps: list[float]) -> tuple[DiscordClient,
                                                                              list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return responses.pop(0)

    client = DiscordClient("secret-token", transport=httpx.MockTransport(handler),
                           sleep=sleeps.append)
    return client, requests


def test_posts_with_bot_auth_and_returns_id() -> None:
    sleeps: list[float] = []
    client, requests = client_for([httpx.Response(200, json={"id": "111"})], sleeps)
    assert client.post_message("42", PAYLOAD) == "111"
    [request] = requests
    assert request.url == "https://discord.com/api/v10/channels/42/messages"
    assert request.headers["Authorization"] == "Bot secret-token"
    assert sleeps == []


def test_retries_once_on_429() -> None:
    sleeps: list[float] = []
    client, requests = client_for(
        [httpx.Response(429, json={"retry_after": 1.5}), httpx.Response(200, json={"id": "7"})],
        sleeps,
    )
    assert client.post_message("42", PAYLOAD) == "7"
    assert len(requests) == 2
    assert sleeps == [1.5]


def test_retry_after_is_capped_and_header_fallback_used() -> None:
    sleeps: list[float] = []
    client, _ = client_for(
        [httpx.Response(429, headers={"Retry-After": "900"}, text="slow down"),
         httpx.Response(200, json={"id": "7"})],
        sleeps,
    )
    client.post_message("42", PAYLOAD)
    assert sleeps == [30.0]


def test_waits_for_bucket_reset_before_next_post() -> None:
    sleeps: list[float] = []
    exhausted = {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset-After": "1.25"}
    client, requests = client_for(
        [httpx.Response(200, headers=exhausted, json={"id": "1"}),
         httpx.Response(200, headers={"X-RateLimit-Remaining": "4"}, json={"id": "2"}),
         httpx.Response(200, json={"id": "3"})],
        sleeps,
    )
    assert [client.post_message("42", PAYLOAD) for _ in range(3)] == ["1", "2", "3"]
    # Only the post after the exhausted bucket waits, and it waits exactly once.
    assert sleeps == [1.25]


def test_no_wait_while_bucket_has_requests_left() -> None:
    sleeps: list[float] = []
    client, _ = client_for(
        [httpx.Response(200, headers={"X-RateLimit-Remaining": "3",
                                      "X-RateLimit-Reset-After": "2"}, json={"id": "1"}),
         httpx.Response(200, json={"id": "2"})],
        sleeps,
    )
    client.post_message("42", PAYLOAD)
    client.post_message("42", PAYLOAD)
    assert sleeps == []


def test_second_429_raises() -> None:
    client, requests = client_for(
        [httpx.Response(429, json={"retry_after": 0}), httpx.Response(429, json={"retry_after": 0})],
        [],
    )
    with pytest.raises(DiscordError, match="HTTP 429"):
        client.post_message("42", PAYLOAD)
    assert len(requests) == 2


def test_other_errors_raise_without_retry_or_token() -> None:
    client, requests = client_for([httpx.Response(403, json={"message": "Missing Access"})], [])
    with pytest.raises(DiscordError, match="HTTP 403") as excinfo:
        client.post_message("42", PAYLOAD)
    assert len(requests) == 1
    assert "secret-token" not in str(excinfo.value)


def test_network_error_is_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = DiscordClient("t", transport=httpx.MockTransport(handler), sleep=lambda _: None)
    with pytest.raises(DiscordError, match="ConnectError"):
        client.post_message("42", PAYLOAD)
