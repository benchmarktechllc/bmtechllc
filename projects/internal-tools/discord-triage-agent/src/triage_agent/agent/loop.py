"""The hand-written tool-use loop (SPEC.md section 2, step 3) and its PreToolUse gate
(SPEC.md section 5).

It continues only while stop_reason == "tool_use". Every model call appends the assistant
message, writes traces and bumps the heartbeat. Every tool_use block goes through the gate
first: allowed calls run, denied calls get an error result, and a call that needs approval is
not executed. If a turn has any of those, the loop pauses: it records one pending approval per
call and returns `awaiting_approval` without appending the turn's results (agent/resume.py
appends them once every request is decided). The loop never talks to Discord: it returns one
LoopOutcome, and the caller acts on it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any, Final, Literal, cast
from uuid import UUID

import anthropic
import psycopg
from anthropic.types import Message, MessageParam, ToolUseBlock

from triage_agent.agent.classifier import Classification
from triage_agent.agent.client import ModelClient, ModelRequest
from triage_agent.agent.prompts import agent_system_prompt
from triage_agent.config import DEFAULT_APPROVAL_TTL
from triage_agent.db.approvals import create_pending
from triage_agent.db.conversation import (
    HookDecision,
    JsonObject,
    StoredMessage,
    Trace,
    append_message,
    insert_trace,
    load_messages,
)
from triage_agent.db.sessions import set_status, touch_heartbeat
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


def waiting_note(session_id: UUID, pending: int) -> str:
    return f"Waiting for approval on {pending} action(s) — session {session_id}"


@dataclass(frozen=True)
class LoopOutcome:
    status: Literal["completed", "failed", "awaiting_approval"]
    reply: str  # the answer, the fallback, or (awaiting_approval) the waiting note
    reason: str  # end_turn | max_tokens | refusal | api_error | iteration_cap | ... | paused
    pending: tuple[UUID, ...] = ()  # approval requests created by the pause, in call order


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
    conn: psycopg.AsyncConnection[Any], block: ToolUseBlock, decision: HookDecision
) -> ToolOutcome:
    if decision != "allow":
        return ToolOutcome(f"tool not permitted: {block.name}", is_error=True)
    tool_input = block.input if isinstance(block.input, dict) else {}
    return await execute_tool(conn, block.name, tool_input)


def tool_result_block(tool_use_id: str, outcome: ToolOutcome) -> JsonObject:
    result: JsonObject = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
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
    *,
    approval_ttl: timedelta = DEFAULT_APPROVAL_TTL,
) -> LoopOutcome:
    await append_message(conn, session_id, "user", [{"type": "text", "text": question}])
    return await continue_loop(
        conn, model, session_id, classification, start_iteration=1, approval_ttl=approval_ttl
    )


async def continue_loop(
    conn: psycopg.AsyncConnection[Any],
    model: ModelClient,
    session_id: UUID,
    classification: Classification,
    *,
    start_iteration: int,
    approval_ttl: timedelta = DEFAULT_APPROVAL_TTL,
) -> LoopOutcome:
    """Run the loop from the stored messages, whose last one must be a user message. A resume
    passes the next iteration number, so the 8-call cap counts the whole session."""
    system = agent_system_prompt(classification.concerns, classification.more_concerns)

    for iteration in range(start_iteration, MAX_ITERATIONS + 1):
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
            blocked: list[ToolUseBlock] = []
            for block in tool_uses:
                # The gate: plain code, decided by the tool name alone.
                decision = decide(block.name)
                call = replace(
                    base,
                    tool_use_id=block.id,
                    tool_name=block.name,
                    tool_input=block.input,
                    hook_decision=decision,
                )
                if decision == "require_approval":
                    blocked.append(block)
                    traces.append(call)  # no result until a human decides
                    continue
                outcome = await _run_tool(conn, block, decision)
                results.append(tool_result_block(block.id, outcome))
                traces.append(
                    replace(
                        call,
                        tool_result={"content": outcome.content, "is_error": outcome.is_error},
                    )
                )
            if blocked:
                pending = await _pause(conn, session_id, iteration, traces, blocked, approval_ttl)
                return LoopOutcome(
                    status="awaiting_approval",
                    reply=waiting_note(session_id, len(pending)),
                    reason="paused",
                    pending=pending,
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


async def _pause(
    conn: psycopg.AsyncConnection[Any],
    session_id: UUID,
    iteration: int,
    traces: list[Trace],
    blocked: list[ToolUseBlock],
    approval_ttl: timedelta,
) -> tuple[UUID, ...]:
    """One transaction: the turn's traces, one pending request per blocked call, and the
    session set to awaiting_approval. The turn's results are not appended yet."""
    async with conn.transaction():
        for trace in traces:
            await insert_trace(conn, trace)
        cur = await conn.execute(
            "SELECT asker_user_id FROM sessions WHERE id = %s FOR UPDATE", (session_id,)
        )
        row = await cur.fetchone()
        if row is None:
            raise LookupError(f"session {session_id} not found")
        pending = [
            await create_pending(
                conn,
                session_id=session_id,
                turn=iteration,
                tool_use_id=block.id,
                tool_name=block.name,
                tool_input=block.input,
                requested_by=row[0],
                ttl=approval_ttl,
            )
            for block in blocked
        ]
        await set_status(conn, session_id, "awaiting_approval")
    return tuple(pending)
