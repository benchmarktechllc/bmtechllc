"""Budget guard for live runs, and the live smoke fixtures' shape."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from triage_agent.agent.classifier import CATEGORIES
from triage_agent.budget import (
    BUDGET_LIMIT_USD,
    PRICES,
    PRICES_CHECKED,
    call_cost,
    can_start_session,
    session_ceiling_usd,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "live_cases.json"


def test_limit_is_four_dollars() -> None:
    assert BUDGET_LIMIT_USD == 4.0


def test_prices_are_dated_and_cover_both_models() -> None:
    assert PRICES_CHECKED
    assert set(PRICES) == {"claude-sonnet-5-5", "claude-haiku-5-5"}


def test_call_cost_uses_per_million_prices() -> None:
    sonnet = PRICES["claude-sonnet-5-5"]
    assert call_cost("claude-sonnet-5-5", 1_000_000, 1_000_000) == pytest.approx(
        sonnet.input_per_mtok + sonnet.output_per_mtok
    )
    assert call_cost("claude-haiku-5-5", 0, 0) == 0


def test_call_cost_rejects_an_unpriced_model() -> None:
    with pytest.raises(KeyError):
        call_cost("claude-opus-5-5", 1, 1)


def test_ceiling_covers_eight_sonnet_calls_at_the_output_cap() -> None:
    sonnet = PRICES["claude-sonnet-5-5"]
    assert session_ceiling_usd() >= 8 * 1024 * sonnet.output_per_mtok / 1_000_000


def test_guard_refuses_a_session_whose_ceiling_would_pass_the_limit() -> None:
    ceiling = session_ceiling_usd()
    assert can_start_session(spent_usd=0.0) is True
    assert can_start_session(spent_usd=BUDGET_LIMIT_USD - ceiling) is True
    assert can_start_session(spent_usd=BUDGET_LIMIT_USD - ceiling + 0.01) is False
    assert can_start_session(spent_usd=3.99) is False


def test_live_fixtures_are_well_formed() -> None:
    cases = json.loads(FIXTURES.read_text())
    assert 1 <= len(cases) <= 5
    for case in cases:
        assert case["question"].strip()
        assert case["expected_concerns"]
        assert set(case["expected_categories"]) <= set(CATEGORIES)
    multi = [
        c
        for c in cases
        if len({t for t in ("T-1" + str(n).zfill(3) for n in range(1, 11)) if t in c["question"]})
        == 2
        and "how_to" in c["expected_categories"]
    ]
    assert multi, "one fixture asks about two synthetic tickets plus one how-to topic"
    assert len(multi[0]["expected_concerns"]) >= 3
