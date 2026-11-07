"""Adversarial checks for classifiers, headcount, roster, and Discord retention."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from benchmark_cli import agent
from benchmark_cli.acceptance import _check, _outcome
from benchmark_cli.discord_sync import plan_discord_reconcile
from benchmark_cli.grounding import (
    asks_for_core_team,
    asks_for_employee_count,
    asks_for_portal_link,
    asks_for_synced_channels,
    stated_employee_count_answer,
)
from benchmark_cli.ingestion.clean import discord_message_is_authoritative
from benchmark_cli.models import AnswerResponse, Citation
from benchmark_cli.retrieval import HybridSearchResult, RetrievalDecision
from benchmark_cli.storage.postgres import DiscordSourceMessage, SearchChunk

_DIGEST = "b" * 64
_HOME = "https://benchmarktechllc.github.io/bmtechllc/portal/"


def _result(text: str) -> HybridSearchResult:
    return HybridSearchResult(
        SearchChunk(3, "discord-channel-1.json", 0, text, {"source_type": "discord"}, 1.0, content_sha256=_DIGEST),
        0.1,
        vector_score=0.8,
        fts_score=1.0,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "question",
    (
        "Where is the deployment guide in the knowledge base?",
        "What did Discord channel #ops decide about deploy?",
        "What is the current employee handbook?",
        "What did the core team decide about pricing?",
    ),
)
def test_ordinary_prompts_are_not_intercepted(question: str) -> None:
    assert asks_for_portal_link(question) is False
    assert asks_for_synced_channels(question) is False
    assert asks_for_employee_count(question) is False
    assert asks_for_core_team(question) is False


@pytest.mark.unit
def test_catalog_questions_still_match() -> None:
    assert asks_for_portal_link("What is the link to the knowledge base portal?")
    assert asks_for_portal_link("Where do I open the knowledge base?")
    assert asks_for_portal_link("What's the URL for our KB?")
    assert asks_for_employee_count("How many employees does BMTech currently have?")
    assert asks_for_employee_count("What is our current headcount?")
    assert asks_for_core_team("Who are the core team members at BMTech?")
    assert asks_for_core_team("Who is on the core team?")
    assert asks_for_synced_channels("Which Discord channels are synced?")
    assert asks_for_synced_channels("What Discord channels do we ingest?")


@pytest.mark.unit
def test_negated_and_third_party_counts_are_not_the_company_headcount() -> None:
    denied = _result("BMTech does not have 12 employees.")
    client = _result("The client has 50 employees.")
    handbook = _result("The employee handbook mentions 4 employees in an example.")
    affirmed = _result("BMTech currently has 12 employees.")

    assert stated_employee_count_answer((denied, client, handbook)) is None
    stated = stated_employee_count_answer((affirmed,))
    assert stated is not None
    assert stated.citations[0].quote == "BMTech currently has 12 employees."


@pytest.mark.unit
def test_heading_and_denial_are_not_a_core_team_roster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "call_llm", lambda **_kwargs: pytest.fail("roster extractor must run first"))
    monkeypatch.setattr(agent, "retrieve_evidence", lambda *_a, **_k: RetrievalDecision(
        "accepted",
        (
            _result("Core team"),
            _result("We do not have a core team."),
        ),
        0.8,
        2,
    ))

    response = agent.run_agent("Who is on the core team?")

    assert response.status == "no_relevant_information"
    assert response.citations == []
    assert "does not name a core team" in response.answer.casefold()
    assert "Ada" not in response.answer


@pytest.mark.unit
def test_answered_status_alone_does_not_prove_employee_or_roster_facts() -> None:
    uncertain = {
        "status": "answered",
        "answer": "The retrieved evidence does not state a current employee count. Ada is a collaborator.",
        "citations": [{"source_file": "discord-channel-1.json", "chunk_id": 1, "quote": "Ada collaborates."}],
    }
    denial = {
        "status": "answered",
        "answer": "The retrieved evidence does not name a core team.",
        "citations": [{"source_file": "notes.md", "chunk_id": 2, "quote": "Core team"}],
    }
    employee = _outcome("How many employees does BMTech currently have?", uncertain, [])
    roster = _outcome("Who are the core team members at BMTech?", denial, [])
    assert employee == {"outcome": "unknown", "proves_fact": False, "expectation": "unknown_ok"}
    assert roster["proves_fact"] is False
    assert roster["outcome"] == "unknown"
    assert _check("How many employees does BMTech currently have?", uncertain, home=_HOME) == []


@pytest.mark.unit
def test_reconcile_keeps_human_messages_and_drops_only_exact_bot_templates() -> None:
    sent = datetime(2026, 11, 1, tzinfo=UTC)
    kept = (
        "There is no information available regarding Q3 plan until Friday.",
        "New documentation published for API.",
        "Ship portal fix today.",
    )
    for text in kept:
        assert discord_message_is_authoritative(text, "Marci") is True
    assert discord_message_is_authoritative("New documentation published", "Marci") is False
    assert discord_message_is_authoritative("Ship portal fix today.", "Deploy Bot") is False
    plan = plan_discord_reconcile(
        [
            DiscordSourceMessage("1", "1", "ops", "Marci", kept[0], sent),
            DiscordSourceMessage("1", "2", "ops", "Marci", kept[1], sent),
            DiscordSourceMessage("1", "3", "ops", "Marci", kept[2], sent),
            DiscordSourceMessage("1", "4", "ops", "Marci", "New documentation published", sent),
        ]
    )
    assert plan.deleted_ids == ("4",)
    assert plan.rewritten == ()


@pytest.mark.unit
def test_ordinary_prompt_reaches_the_model_with_redacted_tool_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = "The deployment guide is the runbook page."
    secret = "ghp_" + ("c" * 36)
    chunk = SearchChunk(
        9,
        "portal/runbooks/deploy.md",
        0,
        f"Token {secret}. {evidence}",
        {"source_type": "kb_page", "url": "/runbooks/deploy/"},
        1.0,
        content_sha256=_DIGEST,
    )
    decision = RetrievalDecision(
        "accepted",
        (HybridSearchResult(chunk, 0.2, vector_score=0.9, fts_score=1.0),),
        0.9,
        1,
    )
    seen: list[str] = []

    def fake_llm(**kwargs: object) -> SimpleNamespace:
        seen.append(str(kwargs))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=""))])

    def fake_structured(**_kwargs: object) -> AnswerResponse:
        return AnswerResponse(
            status="answered",
            answer="The deployment guide is the runbook page.",
            confidence=0.8,
            citations=[
                Citation(
                    source_file=chunk.source_file,
                    chunk_id=chunk.id,
                    content_sha256=_DIGEST,
                    quote=evidence,
                )
            ],
        )

    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "classify_query", lambda _question: "cheap")
    monkeypatch.setattr(agent, "retrieve_evidence", lambda *_a, **_k: decision)
    monkeypatch.setattr(agent, "call_llm", fake_llm)
    monkeypatch.setattr(agent, "call_llm_structured", fake_structured)

    response = agent.run_agent("Where is the deployment guide in the knowledge base?")

    assert seen
    assert secret not in seen[0]
    assert "[REDACTED]" in seen[0]
    assert response.answer == evidence
    assert _HOME not in response.answer


@pytest.mark.unit
def test_tool_results_redact_secrets_before_they_reach_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "ghp_" + ("d" * 36)
    chunk = SearchChunk(
        4,
        "discord-channel-1.json",
        0,
        f"Announcement {secret} stays in the database row.",
        {"source_type": "discord"},
        1.0,
        content_sha256=_DIGEST,
    )
    decision = RetrievalDecision(
        "accepted",
        (HybridSearchResult(chunk, 0.1, vector_score=0.8, fts_score=1.0),),
        0.8,
        1,
    )
    monkeypatch.setattr(agent, "retrieve_evidence", lambda *_a, **_k: decision)
    evidence: dict[int, SearchChunk] = {}

    payload = agent._execute_tool("search_docs", {"query": "announcement"}, evidence)

    assert secret not in payload
    assert "[REDACTED]" in payload
    assert secret not in evidence[4].chunk_text
