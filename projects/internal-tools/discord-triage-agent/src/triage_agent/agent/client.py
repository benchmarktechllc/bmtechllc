"""The one seam between our code and the Anthropic API.

Everything that calls a model takes a `ModelClient`. Production uses `AnthropicModelClient`;
tests use the scripted fake in tests/conftest.py, so no test ever makes a live call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Protocol

from anthropic import AsyncAnthropic, omit
from anthropic.types import (
    Message,
    MessageParam,
    OutputConfigParam,
    ThinkingConfigParam,
    ToolParam,
)

# Every Anthropic call must finish within 2 min (SPEC.md section 2). The SDK retries a timed-out
# request, so the worst case is timeout x (retries + 1) plus a short backoff: 45 x 2 = 90 s.
ANTHROPIC_TIMEOUT_S: Final = 45.0
ANTHROPIC_MAX_RETRIES: Final = 1


@dataclass(frozen=True)
class ModelRequest:
    """The subset of a Messages API request this app sends. No server-side `fallbacks`: a
    refusal must end the session, and every call must run on the named model (DECISIONS.md)."""

    model: str
    max_tokens: int
    messages: list[MessageParam]
    system: str | None = None
    tools: list[ToolParam] = field(default_factory=list)
    output_config: OutputConfigParam | None = None
    thinking: ThinkingConfigParam | None = None


class ModelClient(Protocol):
    async def create(self, request: ModelRequest) -> Message: ...


def make_sdk_client(api_key: str | None) -> AsyncAnthropic:
    return AsyncAnthropic(
        api_key=api_key, timeout=ANTHROPIC_TIMEOUT_S, max_retries=ANTHROPIC_MAX_RETRIES
    )


class AnthropicModelClient:
    def __init__(self, api_key: str | None, sdk: AsyncAnthropic | None = None) -> None:
        self._sdk = sdk or make_sdk_client(api_key)

    async def create(self, request: ModelRequest) -> Message:
        return await self._sdk.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            messages=request.messages,
            system=request.system if request.system is not None else omit,
            tools=request.tools if request.tools else omit,
            output_config=request.output_config if request.output_config is not None else omit,
            thinking=request.thinking if request.thinking is not None else omit,
        )
