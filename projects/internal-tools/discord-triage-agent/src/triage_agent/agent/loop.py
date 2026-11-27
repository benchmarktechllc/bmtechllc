"""The hand-written tool-use loop (SPEC.md section 2, step 3).

It continues only while stop_reason == "tool_use". Every model call appends the assistant
message, writes traces and bumps the heartbeat. The loop never talks to Discord: it returns one
LoopOutcome, and the caller sends exactly one reply from it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Final, Literal, cast
from uuid import UUID

import anthropic
import psycopg
from anthropic.types import Message, MessageParam, ToolUseBlock

from triage_agent.agent.classifier import Classification
from triage_agent.agent.client import ModelClient, ModelRequest
from triage_agent.agent.prompts import agent_system_prompt
from triage_agent.db.conversation import (
    JsonObject,
    StoredMessage,
    Trace,
    append_message,
    insert_trace,
    load_messages,
)
from triage_agent.db.sessions import touch_heartbeat
from triage_agent.tools.definitions import TOOLS
from triage_agent.tools.executors import ToolOutcome, execute_tool
from triage_agent.tools.policy import decide

AGENT_MODEL: Final = "claude-sonnet-5-5"
AGENT_MAX_TOKENS: Final = 1024
AGENT_EFFORT: Final = "low"
MAX_ITERATIONS: Final = 8


def fallback_reply(session_id: UUID) -> str:
    return (
        "Sorry, I couldn't finish answering this question. Please try again, or ask a "
        f"teammate. (session {session_id})"
    )


@dataclass(frozen=True)
class LoopOutcome:
    status: Literal["completed", "failed"]
    reply: str
    reason: str  # end_turn | max_tokens | refusal | api_error | iteration_cap | empty_reply | ...


def _failed(session_id: UUID, reason: str) -> LoopOutcome:
    return LoopOutcome(status="failed", reply=fallback_reply(session_id), reason=reason)


def _as_param(m: StoredMessage) -> MessageParam:
    # Stored content is the exact API content blocks we appended, so it round-trips as-is.
    return cast(MessageParam, {"role": m.role, "content": m.content})


def _dump_content(response: Message) -> list[JsonObject]:
    # Thinking blocks (with signatures) are kept and sent back unchanged.
    return [b.model_dump(mode="json", exclude_none=True) for b in response.content]


def _ms_since(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


async def _run_tool(
    conn: psycopg.AsyncConnection[Any], block: ToolUseBlock
) -> tuple[ToolOutcome, Literal["allow", "require_approval", "deny"]]:
    decision = decide(block.name)
    if decision != "allow":
        return ToolOutcome(f"tool not permitted: {block.name}", is_error=True), decision
    tool_input = block.input if isinstance(block.input, dict) else {}
    return await execute_tool(conn, block.name, tool_input), decision


def _tool_result_block(block: ToolUseBlock, outcome: ToolOutcome) -> JsonObject:
    result: JsonObject = {
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": outcome.content,
    }
    if outcome.is_error:
        result["is_error"] = True
    return result


async def run_loop(
    conn: psycopg.AsyncConnection[Any],
    model: ModelClient,
    session_id: UUID,
    question: str,
    classification: Classification,
) -> LoopOutcome:
    system = agent_system_prompt(classification.concerns, classification.more_concerns)
    await append_message(conn, session_id, "user", [{"type": "text", "text": question}])

    for iteration in range(1, MAX_ITERATIONS + 1):
        # The request is rebuilt from stored rows, so what the model sees is what is stored.
        history = await load_messages(conn, session_id)
        request = ModelRequest(
            model=AGENT_MODEL,
            max_tokens=AGENT_MAX_TOKENS,
            system=system,
            tools=list(TOOLS),
            messages=[_as_param(m) for m in history],
            output_config={"effort": AGENT_EFFORT},
        )
        start = time.monotonic()
        try:
            response = await model.create(request)
        except anthropic.APIError as exc:
            async with conn.transaction():
                await insert_trace(
                    conn,
                    Trace(
                        session_id=session_id,
                        iteration=iteration,
                        model=AGENT_MODEL,
                        latency_ms=_ms_since(start),
                        error=f"{type(exc).__name__}: {exc}"[:500],
                    ),
                )
                await touch_heartbeat(conn, session_id, iterations=iteration)
            return _failed(session_id, "api_error")
        latency_ms = _ms_since(start)

        base = Trace(
            session_id=session_id,
            iteration=iteration,
            model=response.model,
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
        )
        async with conn.transaction():
            await append_message(conn, session_id, "assistant", _dump_content(response))
            await touch_heartbeat(conn, session_id, iterations=iteration)

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason == "tool_use" and tool_uses:
            traces: list[Trace] = []
            results: list[JsonObject] = []
            for block in tool_uses:
                outcome, decision = await _run_tool(conn, block)
                results.append(_tool_result_block(block, outcome))
                traces.append(
                    replace(
                        base,
                        tool_use_id=block.id,
                        tool_name=block.name,
                        tool_input=block.input,
                        tool_result={"content": outcome.content, "is_error": outcome.is_error},
                        hook_decision=decision,
                    )
                )
            # All of the turn's results go back together, in one user message.
            async with conn.transaction():
                for trace in traces:
                    await insert_trace(conn, trace)
                await append_message(conn, session_id, "user", results)
                await touch_heartbeat(conn, session_id)
            continue

        await insert_trace(conn, base)
        if response.stop_reason == "end_turn":
            text = "\n\n".join(b.text for b in response.content if b.type == "text").strip()
            if not text:
                return _failed(session_id, "empty_reply")
            return LoopOutcome(status="completed", reply=text, reason="end_turn")
        # max_tokens, refusal, or anything else unexpected: never send a partial answer.
        return _failed(session_id, response.stop_reason or "no_stop_reason")

    return _failed(session_id, "iteration_cap")
