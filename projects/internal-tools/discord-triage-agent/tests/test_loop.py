"""The agent loop, end to end through handle_ask, with the scripted fake Anthropic client,
the mocked Discord REST client and a seeded local Postgres."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import httpx
import psycopg
import pytest
from anthropic.types import Message

from tests.conftest import (
    FakeAnthropic,
    FakeDiscordRest,
    api_error,
    api_timeout,
    classifier_reply,
    insert_test_session,
    make_message,
    text_reply,
    tool_call,
    tool_turn,
)
from triage_agent.agent.client import (
    ANTHROPIC_MAX_RETRIES,
    ANTHROPIC_TIMEOUT_S,
    ModelRequest,
    make_sdk_client,
)
from triage_agent.agent.loop import (
    AGENT_MAX_TOKENS,
    AGENT_MODEL,
    MAX_ITERATIONS,
    fallback_reply,
)
from triage_agent.agent.pipeline import AskJob, handle_ask
from triage_agent.agent.prompts import OTHER_CONCERNS_INSTRUCTION

Db = psycopg.Connection[tuple[Any, ...]]

ONE_CONCERN = classifier_reply([("status of ticket T-1002", "billing")])


@dataclass
class Run:
    session_id: UUID
    model: FakeAnthropic
    discord: FakeDiscordRest

    @property
    def agent_requests(self) -> list[ModelRequest]:
        return [r for r in self.model.requests if r.model == AGENT_MODEL]

    @property
    def replies(self) -> list[str]:
        return [e[2] for e in self.discord.edits]


async def run_ask(
    database_url: str,
    db: Db,
    script: list[Message | Exception],
    question: str = "what's the status of T-1002?",
    model: FakeAnthropic | None = None,
    discord: FakeDiscordRest | None = None,
) -> Run:
    session_id = insert_test_session(db, question)
    model = model or FakeAnthropic()
    model.script = list(script)
    discord = discord or FakeDiscordRest()
    await handle_ask(
        database_url,
        model,
        discord,
        AskJob(session_id=session_id, question=question, application_id="app", token="tok"),
    )
    assert model.script == [], "every scripted response should have been consumed"
    return Run(session_id, model, discord)


def session_row(db: Db, session_id: UUID) -> tuple[Any, ...]:
    row = db.execute(
        "SELECT status, iterations, reply_error FROM sessions WHERE id = %s", (session_id,)
    ).fetchone()
    assert row is not None
    return row


def message_rows(db: Db, session_id: UUID) -> list[tuple[int, str, Any]]:
    return [
        (r[0], r[1], r[2])
        for r in db.execute(
            "SELECT seq, role, content FROM messages WHERE session_id = %s ORDER BY seq",
            (session_id,),
        )
    ]


def agent_traces(db: Db, session_id: UUID) -> list[tuple[Any, ...]]:
    return db.execute(
        "SELECT iteration, model, stop_reason, input_tokens, output_tokens, tool_use_id,"
        " tool_name, tool_input, tool_result, hook_decision, error"
        " FROM traces WHERE session_id = %s AND iteration > 0 ORDER BY id",
        (session_id,),
    ).fetchall()


# --- the happy paths ---------------------------------------------------------------


@pytest.mark.anyio
async def test_two_tool_run_to_end_turn(seeded_database_url: str, db: Db) -> None:
    run = await run_ask(
        seeded_database_url,
        db,
        [
            ONE_CONCERN,
            tool_turn(tool_call("get_ticket", {"ticket_id": "T-1002"}, "tu_1")),
            tool_turn(tool_call("search_kb", {"query": "refund"}, "tu_2")),
            text_reply("T-1002 is open; refunds take 5–7 days."),
        ],
    )

    assert run.replies == ["T-1002 is open; refunds take 5–7 days."]
    assert session_row(db, run.session_id) == ("completed", 3, None)
    assert len(run.agent_requests) == 3

    # The tool results the model saw are the real executor output from the seeded DB.
    second = run.agent_requests[1].messages[-1]
    assert second["role"] == "user"
    (result,) = list(second["content"])  # type: ignore[arg-type]
    assert isinstance(result, dict) and result["type"] == "tool_result"
    assert result["tool_use_id"] == "tu_1"
    assert "T-1002" in str(result["content"])
    assert not result.get("is_error")

    roles = [r[1] for r in message_rows(db, run.session_id)]
    assert roles == ["user", "assistant", "user", "assistant", "user", "assistant"]


@pytest.mark.anyio
async def test_first_turn_end_turn_makes_one_model_call_and_one_reply(
    seeded_database_url: str, db: Db
) -> None:
    run = await run_ask(seeded_database_url, db, [ONE_CONCERN, text_reply("Hello!")])
    assert len(run.agent_requests) == 1
    assert run.replies == ["Hello!"]
    assert session_row(db, run.session_id) == ("completed", 1, None)
    traces = agent_traces(db, run.session_id)
    assert len(traces) == 1
    iteration, model, stop_reason, *_, tool_name, _ti, _tr, hook, error = traces[0]
    assert (iteration, model, stop_reason, tool_name, hook, error) == (
        1,
        AGENT_MODEL,
        "end_turn",
        None,
        None,
        None,
    )


@pytest.mark.anyio
async def test_parallel_tool_calls_are_answered_in_one_user_message(
    seeded_database_url: str, db: Db
) -> None:
    run = await run_ask(
        seeded_database_url,
        db,
        [
            ONE_CONCERN,
            tool_turn(
                tool_call("get_ticket", {"ticket_id": "T-1002"}, "tu_a"),
                tool_call("get_ticket", {"ticket_id": "T-1005"}, "tu_b"),
                tool_call("search_kb", {"query": "password reset"}, "tu_c"),
            ),
            text_reply("Done."),
        ],
    )
    follow_up = run.agent_requests[1].messages
    # [question, assistant tool turn, ONE user message with all three results]
    assert [m["role"] for m in follow_up] == ["user", "assistant", "user"]
    results = list(follow_up[-1]["content"])  # type: ignore[arg-type]
    assert [r["tool_use_id"] for r in results] == ["tu_a", "tu_b", "tu_c"]  # type: ignore[index]
    assert all(r["type"] == "tool_result" for r in results)  # type: ignore[index]

    traces = agent_traces(db, run.session_id)
    tool_rows = [t for t in traces if t[6] is not None]
    assert [(t[0], t[5], t[6], t[9]) for t in tool_rows] == [
        (1, "tu_a", "get_ticket", "allow"),
        (1, "tu_b", "get_ticket", "allow"),
        (1, "tu_c", "search_kb", "allow"),
    ]
    assert tool_rows[0][7] == {"ticket_id": "T-1002"}
    assert "T-1002" in json.dumps(tool_rows[0][8])


@pytest.mark.anyio
async def test_multi_concern_question_gets_a_single_reply_and_the_checklist(
    seeded_database_url: str, db: Db
) -> None:
    concerns = [
        ("refund status on T-1002", "billing"),
        ("crash reported in T-1005", "bug"),
        ("how to reset a password", "how_to"),
    ]
    run = await run_ask(
        seeded_database_url,
        db,
        [
            classifier_reply(concerns, primary=0),
            tool_turn(
                tool_call("get_ticket", {"ticket_id": "T-1002"}),
                tool_call("get_ticket", {"ticket_id": "T-1005"}),
                tool_call("search_kb", {"query": "reset password"}),
            ),
            text_reply("All three answered."),
        ],
        question="Where's my refund on T-1002, is the T-1005 crash fixed, and how do I reset "
        "my password?",
    )
    assert run.replies == ["All three answered."]
    system = run.agent_requests[0].system or ""
    for summary, _ in concerns:
        assert summary in system
    assert OTHER_CONCERNS_INSTRUCTION not in system
    assert db.execute(
        "SELECT categories FROM sessions WHERE id = %s", (run.session_id,)
    ).fetchone() == (["billing", "bug", "how_to"],)


@pytest.mark.anyio
async def test_more_concerns_adds_the_answer_any_other_concern_instruction(
    seeded_database_url: str, db: Db
) -> None:
    run = await run_ask(
        seeded_database_url,
        db,
        [
            classifier_reply(
                [("a", "billing"), ("b", "bug"), ("c", "account")], more_concerns=True
            ),
            text_reply("ok"),
        ],
    )
    assert OTHER_CONCERNS_INSTRUCTION in (run.agent_requests[0].system or "")


@pytest.mark.anyio
async def test_low_confidence_checklist_is_the_question_as_other(
    seeded_database_url: str, db: Db
) -> None:
    question = "hmm T-1002??"
    run = await run_ask(
        seeded_database_url,
        db,
        [classifier_reply([("x", "billing")], confidence=0.59), text_reply("ok")],
        question=question,
    )
    assert question in (run.agent_requests[0].system or "")


@pytest.mark.anyio
async def test_agent_request_shape(seeded_database_url: str, db: Db) -> None:
    run = await run_ask(seeded_database_url, db, [ONE_CONCERN, text_reply("ok")])
    (req,) = run.agent_requests
    assert req.model == "claude-sonnet-5-5"
    assert req.max_tokens == AGENT_MAX_TOKENS == 1024
    assert req.output_config == {"effort": "low"}
    assert sorted(t["name"] for t in req.tools) == ["close_ticket", "get_ticket", "search_kb"]
    assert req.messages[0]["role"] == "user"
    assert "what's the status of T-1002?" in json.dumps(req.messages[0]["content"])


@pytest.mark.anyio
async def test_unknown_tool_is_denied_with_an_error_result(
    seeded_database_url: str, db: Db
) -> None:
    run = await run_ask(
        seeded_database_url,
        db,
        [
            ONE_CONCERN,
            tool_turn(tool_call("delete_ticket", {"ticket_id": "T-1002"}, "tu_x")),
            text_reply("I can't delete tickets."),
        ],
    )
    (result,) = list(run.agent_requests[1].messages[-1]["content"])  # type: ignore[arg-type]
    assert result["is_error"] is True  # type: ignore[index]
    (row,) = [t for t in agent_traces(db, run.session_id) if t[6] == "delete_ticket"]
    assert row[9] == "deny"
    assert db.execute("SELECT status FROM tickets WHERE id = 'T-1002'").fetchone() == ("open",)


@pytest.mark.anyio
async def test_thinking_blocks_are_stored_and_sent_back_unchanged(
    seeded_database_url: str, db: Db
) -> None:
    thinking = {"type": "thinking", "thinking": "", "signature": "sig-abc"}
    run = await run_ask(
        seeded_database_url,
        db,
        [
            ONE_CONCERN,
            make_message(
                [thinking, tool_call("get_ticket", {"ticket_id": "T-1002"})],
                stop_reason="tool_use",
            ),
            text_reply("ok"),
        ],
    )
    sent = run.agent_requests[1].messages[1]
    assert list(sent["content"])[0] == thinking  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_reply_is_capped_at_discords_2000_characters(
    seeded_database_url: str, db: Db
) -> None:
    run = await run_ask(seeded_database_url, db, [ONE_CONCERN, text_reply("x" * 2500)])
    (reply,) = run.replies
    assert len(reply) <= 2000


# --- failures: one fallback reply, session failed -------------------------------------


@pytest.mark.anyio
async def test_iteration_cap_sends_the_fallback_and_fails(seeded_database_url: str, db: Db) -> None:
    turns = [tool_turn(tool_call("search_kb", {"query": f"q{i}"})) for i in range(MAX_ITERATIONS)]
    run = await run_ask(seeded_database_url, db, [ONE_CONCERN, *turns])
    assert MAX_ITERATIONS == 8
    assert len(run.agent_requests) == 8
    assert run.replies == [fallback_reply(run.session_id)]
    assert session_row(db, run.session_id)[:2] == ("failed", 8)


@pytest.mark.anyio
@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
async def test_refusal_and_max_tokens_send_the_fallback_and_fail(
    seeded_database_url: str, db: Db, stop_reason: str
) -> None:
    run = await run_ask(
        seeded_database_url,
        db,
        [
            ONE_CONCERN,
            tool_turn(tool_call("get_ticket", {"ticket_id": "T-1002"})),
            text_reply("partial answer that must not be sent", stop_reason=stop_reason),
        ],
    )
    assert run.replies == [fallback_reply(run.session_id)]
    assert session_row(db, run.session_id)[:2] == ("failed", 2)
    assert agent_traces(db, run.session_id)[-1][2] == stop_reason


@pytest.mark.anyio
@pytest.mark.parametrize("error", [api_error(), api_timeout()], ids=["api_error", "timeout"])
async def test_api_error_and_timeout_send_the_fallback_and_fail(
    seeded_database_url: str, db: Db, error: Exception
) -> None:
    run = await run_ask(
        seeded_database_url,
        db,
        [ONE_CONCERN, tool_turn(tool_call("get_ticket", {"ticket_id": "T-1002"})), error],
    )
    assert run.replies == [fallback_reply(run.session_id)]
    status, iterations, _ = session_row(db, run.session_id)
    assert status == "failed"
    last = agent_traces(db, run.session_id)[-1]
    assert last[0] == 2 and last[2] is None and last[10]  # iteration 2, no stop_reason, error
    # No assistant message is stored for the failed call.
    assert [r[1] for r in message_rows(db, run.session_id)] == ["user", "assistant", "user"]


@pytest.mark.anyio
async def test_empty_end_turn_sends_the_fallback(seeded_database_url: str, db: Db) -> None:
    run = await run_ask(seeded_database_url, db, [ONE_CONCERN, make_message([])])
    assert run.replies == [fallback_reply(run.session_id)]
    assert session_row(db, run.session_id)[0] == "failed"


def test_fallback_reply_names_the_session() -> None:
    sid = UUID("00000000-0000-0000-0000-000000000001")
    assert str(sid) in fallback_reply(sid)


@pytest.mark.anyio
async def test_discord_failure_is_recorded_and_session_still_completes(
    seeded_database_url: str, db: Db
) -> None:
    discord = FakeDiscordRest(fail_with=httpx.ConnectError("discord down"))
    run = await run_ask(
        seeded_database_url, db, [ONE_CONCERN, text_reply("answer")], discord=discord
    )
    status, _, reply_error = session_row(db, run.session_id)
    assert status == "completed"
    assert reply_error and "discord down" in reply_error


# --- audit trail: messages, traces, heartbeat per iteration -----------------------------


@pytest.mark.anyio
async def test_messages_are_append_only_with_continuous_seq(
    seeded_database_url: str, db: Db
) -> None:
    run = await run_ask(
        seeded_database_url,
        db,
        [
            ONE_CONCERN,
            tool_turn(tool_call("get_ticket", {"ticket_id": "T-1002"}, "tu_1")),
            text_reply("final"),
        ],
    )
    rows = message_rows(db, run.session_id)
    assert [r[0] for r in rows] == [0, 1, 2, 3]
    assert rows[1][2] == [
        {
            "type": "tool_use",
            "id": "tu_1",
            "name": "get_ticket",
            "input": {"ticket_id": "T-1002"},
        }
    ]
    assert rows[3][2] == [{"type": "text", "text": "final"}]
    # What was sent to the model on each call equals what is stored.
    for i, req in enumerate(run.agent_requests):
        stored = [{"role": r[1], "content": r[2]} for r in rows[: 1 + 2 * i]]
        assert [dict(m) for m in req.messages] == stored
    with pytest.raises(psycopg.errors.UniqueViolation):
        db.execute(
            "INSERT INTO messages (session_id, seq, role, content) VALUES (%s, 1, 'user', '[]')",
            (run.session_id,),
        )


@pytest.mark.anyio
async def test_trace_rows_and_heartbeat_updates_per_iteration(
    seeded_database_url: str, db: Db
) -> None:
    seen: list[tuple[int, Any]] = []

    def observe(request: ModelRequest) -> None:
        if request.model != AGENT_MODEL:
            return
        # Each test has a fresh database holding only this session.
        row = db.execute("SELECT iterations, heartbeat_at FROM sessions").fetchone()
        assert row is not None
        seen.append((row[0], row[1]))

    run = await run_ask(
        seeded_database_url,
        db,
        [
            ONE_CONCERN,
            tool_turn(tool_call("get_ticket", {"ticket_id": "T-1002"}, "tu_1")),
            tool_turn(
                tool_call("search_kb", {"query": "refund"}, "tu_2"),
                tool_call("get_ticket", {"ticket_id": "T-9999"}, "tu_3"),
            ),
            text_reply("final"),
        ],
        model=FakeAnthropic(on_call=observe),
    )

    # Before agent call N, iterations == N-1, and the heartbeat moved forward each time.
    assert [s[0] for s in seen] == [0, 1, 2]
    beats = [s[1] for s in seen]
    assert beats[0] < beats[1] < beats[2]
    final_beat = db.execute(
        "SELECT heartbeat_at FROM sessions WHERE id = %s", (run.session_id,)
    ).fetchone()
    assert final_beat is not None and final_beat[0] > beats[2]

    traces = agent_traces(db, run.session_id)
    assert [(t[0], t[2], t[6]) for t in traces] == [
        (1, "tool_use", "get_ticket"),
        (2, "tool_use", "search_kb"),
        (2, "tool_use", "get_ticket"),
        (3, "end_turn", None),
    ]
    assert all(t[1] == AGENT_MODEL and t[3] == 100 and t[4] == 20 for t in traces)
    missing = traces[2]
    assert missing[8]["is_error"] is True  # T-9999 is not a ticket


# --- timeouts --------------------------------------------------------------------------


def test_anthropic_calls_are_capped_at_two_minutes_including_retries() -> None:
    sdk = make_sdk_client("sk-test-not-real")
    assert sdk.timeout == ANTHROPIC_TIMEOUT_S
    assert sdk.max_retries == ANTHROPIC_MAX_RETRIES
    assert ANTHROPIC_TIMEOUT_S * (ANTHROPIC_MAX_RETRIES + 1) <= 120
