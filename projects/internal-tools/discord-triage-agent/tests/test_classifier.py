"""Classifier: structured-output call, plain-code validation, and what lands on the session."""

from __future__ import annotations

import json
from typing import Any

import psycopg
import pytest

from tests.conftest import (
    FakeAnthropic,
    api_error,
    classifier_reply,
    insert_test_session,
    make_message,
)
from triage_agent.agent.classifier import (
    CATEGORIES,
    CLASSIFIER_MAX_TOKENS,
    CLASSIFIER_MODEL,
    Classification,
    Concern,
    classify_session,
    parse_classification,
)

QUESTION = "Why was I charged twice, and can I get a refund?"


def raw(body: dict[str, Any]) -> str:
    return json.dumps(body)


def valid_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "concerns": [{"summary": "double charge", "category": "billing"}],
        "primary": 0,
        "confidence": 0.9,
        "more_concerns": False,
    }
    body.update(overrides)
    return body


def assert_other_fallback(c: Classification, question: str = QUESTION) -> None:
    assert c.concerns == (Concern(summary=question, category="other"),)
    assert c.primary_category == "other"
    assert c.categories == ("other",)
    assert c.more_concerns is False


# --- parse_classification (pure) -------------------------------------------------


def test_two_concerns_with_the_same_category_give_one_label() -> None:
    c = parse_classification(
        QUESTION,
        raw(
            valid_body(
                concerns=[
                    {"summary": "charged twice", "category": "billing"},
                    {"summary": "wants a refund", "category": "billing"},
                ],
                primary=1,
            )
        ),
    )
    assert [x.summary for x in c.concerns] == ["charged twice", "wants a refund"]
    assert c.categories == ("billing",)
    assert c.primary_category == "billing"


def test_categories_are_distinct_in_concern_order() -> None:
    c = parse_classification(
        QUESTION,
        raw(
            valid_body(
                concerns=[
                    {"summary": "a", "category": "bug"},
                    {"summary": "b", "category": "billing"},
                    {"summary": "c", "category": "bug"},
                ],
                primary=1,
            )
        ),
    )
    assert c.categories == ("bug", "billing")
    assert c.primary_category == "billing"


def test_four_concern_question_keeps_three_and_sets_more_concerns() -> None:
    c = parse_classification(
        QUESTION,
        raw(
            valid_body(
                concerns=[
                    {"summary": "a", "category": "billing"},
                    {"summary": "b", "category": "account"},
                    {"summary": "c", "category": "how_to"},
                ],
                more_concerns=True,
            )
        ),
    )
    assert len(c.concerns) == 3
    assert c.more_concerns is True


def test_more_than_three_listed_concerns_are_cut_to_three_with_more_concerns() -> None:
    concerns = [{"summary": s, "category": "bug"} for s in "abcd"]
    c = parse_classification(QUESTION, raw(valid_body(concerns=concerns)))
    assert [x.summary for x in c.concerns] == ["a", "b", "c"]
    assert c.more_concerns is True


def test_unknown_category_becomes_other() -> None:
    c = parse_classification(
        QUESTION, raw(valid_body(concerns=[{"summary": "x", "category": "shipping"}]))
    )
    assert_other_fallback(c)


@pytest.mark.parametrize("primary", [1, -1, 3])
def test_out_of_range_primary_is_rejected(primary: int) -> None:
    assert_other_fallback(parse_classification(QUESTION, raw(valid_body(primary=primary))))


def test_confidence_059_becomes_other_but_keeps_the_confidence() -> None:
    c = parse_classification(QUESTION, raw(valid_body(confidence=0.59)))
    assert_other_fallback(c)
    assert c.confidence == pytest.approx(0.59)


def test_confidence_06_is_kept() -> None:
    c = parse_classification(QUESTION, raw(valid_body(confidence=0.6)))
    assert c.primary_category == "billing"


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        raw(valid_body(concerns=[])),
        raw(valid_body(concerns=[{"summary": "", "category": "bug"}])),
        raw(valid_body(confidence=1.5)),
        raw(valid_body(confidence="high")),
        raw(valid_body(primary="0")),
        raw(valid_body(more_concerns="no")),
        raw({"concerns": [{"summary": "x", "category": "bug"}]}),
    ],
)
def test_invalid_output_becomes_other(text: str) -> None:
    c = parse_classification(QUESTION, text)
    assert_other_fallback(c)
    assert c.confidence is None or 0 <= c.confidence <= 1


def test_categories_constant_matches_the_spec() -> None:
    assert CATEGORIES == ("billing", "account", "bug", "how_to", "other")


# --- classify_session (model call + storage) --------------------------------------


@pytest.mark.anyio
async def test_classify_session_stores_the_result_on_the_session(
    database_url: str, db: psycopg.Connection[tuple[Any, ...]]
) -> None:
    session_id = insert_test_session(db, QUESTION)
    model = FakeAnthropic(
        script=[
            classifier_reply(
                [("charged twice", "billing"), ("reset 2FA", "account")],
                primary=0,
                confidence=0.82,
            )
        ]
    )
    async with await psycopg.AsyncConnection.connect(database_url, autocommit=True) as conn:
        result = await classify_session(conn, model, session_id, QUESTION)

    assert result.categories == ("billing", "account")
    row = db.execute(
        "SELECT concerns, more_concerns, primary_category, categories, confidence"
        " FROM sessions WHERE id = %s",
        (session_id,),
    ).fetchone()
    assert row == (
        [
            {"summary": "charged twice", "category": "billing"},
            {"summary": "reset 2FA", "category": "account"},
        ],
        False,
        "billing",
        ["billing", "account"],
        pytest.approx(0.82),
    )


@pytest.mark.anyio
async def test_classifier_request_uses_haiku_structured_output_and_256_tokens(
    database_url: str, db: psycopg.Connection[tuple[Any, ...]]
) -> None:
    session_id = insert_test_session(db, QUESTION)
    model = FakeAnthropic(script=[classifier_reply([("x", "billing")])])
    async with await psycopg.AsyncConnection.connect(database_url, autocommit=True) as conn:
        await classify_session(conn, model, session_id, QUESTION)

    (req,) = model.requests
    assert req.model == CLASSIFIER_MODEL == "claude-haiku-5-5"
    assert req.max_tokens == CLASSIFIER_MAX_TOKENS == 256
    assert req.output_config is not None
    fmt = req.output_config.get("format")
    assert fmt is not None and fmt["type"] == "json_schema"
    schema = fmt["schema"]
    assert set(schema["required"]) == {"concerns", "primary", "confidence", "more_concerns"}
    assert req.tools == []
    # The question is sent as user content, never in the system prompt.
    assert QUESTION in json.dumps(req.messages)
    assert QUESTION not in (req.system or "")


@pytest.mark.anyio
async def test_classifier_call_writes_a_trace_row_and_no_message_row(
    database_url: str, db: psycopg.Connection[tuple[Any, ...]]
) -> None:
    session_id = insert_test_session(db, QUESTION)
    model = FakeAnthropic(script=[classifier_reply([("x", "billing")])])
    async with await psycopg.AsyncConnection.connect(database_url, autocommit=True) as conn:
        await classify_session(conn, model, session_id, QUESTION)

    traces = db.execute(
        "SELECT iteration, model, stop_reason, input_tokens, output_tokens FROM traces"
        " WHERE session_id = %s",
        (session_id,),
    ).fetchall()
    assert traces == [(0, "claude-haiku-5-5", "end_turn", 300, 60)]
    assert db.execute(
        "SELECT count(*) FROM messages WHERE session_id = %s", (session_id,)
    ).fetchone() == (0,)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "response",
    [
        make_message([], stop_reason="refusal", model="claude-haiku-5-5"),
        make_message(
            [{"type": "text", "text": '{"concerns": ['}],
            stop_reason="max_tokens",
            model="claude-haiku-5-5",
        ),
    ],
)
async def test_classifier_refusal_or_truncation_becomes_other(
    database_url: str, db: psycopg.Connection[tuple[Any, ...]], response: Any
) -> None:
    session_id = insert_test_session(db, QUESTION)
    model = FakeAnthropic(script=[response])
    async with await psycopg.AsyncConnection.connect(database_url, autocommit=True) as conn:
        result = await classify_session(conn, model, session_id, QUESTION)
    assert_other_fallback(result)
    assert db.execute(
        "SELECT primary_category, categories FROM sessions WHERE id = %s", (session_id,)
    ).fetchone() == ("other", ["other"])


@pytest.mark.anyio
async def test_classifier_api_error_becomes_other_and_is_traced(
    database_url: str, db: psycopg.Connection[tuple[Any, ...]]
) -> None:
    session_id = insert_test_session(db, QUESTION)
    model = FakeAnthropic(script=[api_error()])
    async with await psycopg.AsyncConnection.connect(database_url, autocommit=True) as conn:
        result = await classify_session(conn, model, session_id, QUESTION)
    assert_other_fallback(result)
    row = db.execute(
        "SELECT stop_reason, error IS NOT NULL FROM traces WHERE session_id = %s", (session_id,)
    ).fetchone()
    assert row == (None, True)


@pytest.mark.anyio
async def test_classifier_prompt_keeps_a_problem_and_its_status_as_one_concern(
    database_url: str, db: psycopg.Connection[tuple[Any, ...]]
) -> None:
    """Live smoke 2026-11-27: "the app crashes on PDFs (T-1005). Is there a fix yet?" was split
    into a `bug` concern and an `other` "is there a fix" concern. The prompt must say a problem
    and the question about its status/fix are one concern with the problem's category, and
    `other` must no longer cover status questions in general."""
    question = "Our iOS app keeps crashing when we open PDF attachments (T-1005). Is there a fix?"
    session_id = insert_test_session(db, question)
    model = FakeAnthropic(script=[classifier_reply([("crash on PDF attachments", "bug")])])
    async with await psycopg.AsyncConnection.connect(database_url, autocommit=True) as conn:
        await classify_session(conn, model, session_id, question)

    system = model.requests[0].system or ""
    assert "one concern" in system
    assert "status, fix or next step" in system
    assert "bare ticket status question" not in system


def test_classifier_prompt_has_a_remediation_example_as_one_concern() -> None:
    """Live smoke 2026-11-27 (2 of 4 runs): "Our webhooks stopped working… What do we need to
    do?" split into bug + how_to. The prompt gives that shape, a broken thing plus "what do we
    do", as an example of one concern in the problem's category, and says not to add a
    separate how_to concern for the fix."""
    from triage_agent.agent.prompts import CLASSIFIER_SYSTEM

    assert "what do we need to do?" in CLASSIFIER_SYSTEM
    assert "do not add a separate how_to concern" in CLASSIFIER_SYSTEM.lower()
