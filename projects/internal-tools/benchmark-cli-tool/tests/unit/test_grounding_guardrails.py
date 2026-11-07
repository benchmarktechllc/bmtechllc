"""Adversarial checks for classifiers, headcount, roster, and Discord retention."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from benchmark_cli import agent
from benchmark_cli.acceptance import _check, _outcome
from benchmark_cli.discord_sync import plan_discord_reconcile
from benchmark_cli.entailment import assess_claim, meaningful_text, supported_steps
from benchmark_cli.grounding import (
    asks_for_core_team,
    asks_for_employee_count,
    asks_for_portal_link,
    asks_for_synced_channels,
    scrub_response,
    stated_employee_count_answer,
)
from benchmark_cli.ingestion.clean import discord_message_is_authoritative, redact_secrets
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
        "What's the URL for the billing page in our KB?",
        "How many staff are on the on-call rotation?",
        "Who did the core team hire?",
        "How do I index a Discord channel?",
        "Send me the link to the portal status page",
        "Please open the portal and tell me who is on call",
        "How many employees does Acme have?",
        "How many employees joined last quarter?",
        "Who should we hire next for the core team?",
        "How do I index a new Discord channel?",
        "Where do I find the kb incident bridge?",
        "Open the kb settings",
        "How do I rotate the portal certificate?",
        "How do I open the portal?",
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
def test_negated_conditional_and_other_party_counts_are_not_headcount() -> None:
    samples = (
        "BMTech hasn't got 12 employees.",
        "BMTech does not have 12 employees.",
        "When the company has 10 employees, file the form.",
        "Acme has 10 employees while BMTech integrates the product.",
        "The client has 50 employees.",
        "BMTech has one employee handbook.",
        "BMTech has 12 employee benefits.",
        "BMTech would have 12 employees.",
        "BMTech might have 12 employees.",
        "BMTech has 12 employees if the plan is approved.",
        "BMTech had 12 employees in 2019.",
        "The company has 10 employees.",
    )
    assert stated_employee_count_answer(tuple(_result(text) for text in samples)) is None
    stated = stated_employee_count_answer((_result("BMTech currently has 12 employees."),))
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
            _result("The core team is meeting tomorrow."),
            _result("Our core team is ready."),
            _result("The core team is Responsible for deployments."),
            _result("The core team members are in Boston."),
            _result("Boston is on the core team."),
            _result("The core team includes Boston."),
        ),
        0.8,
        2,
    ))

    response = agent.run_agent("Who is on the core team?")

    assert response.status == "no_relevant_information"
    assert response.citations == []
    assert "does not name a core team" in response.answer.casefold()
    assert "Ada" not in response.answer
    assert "Boston" not in response.answer


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
    names = ("Marci", "Deploy Bot", "CI", "my-bot", "user_github", "x-ci")
    texts = (
        *kept,
        "New documentation published",
        "Ship portal fix today.",
    )
    for name in names:
        for text in texts:
            assert discord_message_is_authoritative(text, name) is True
    plan = plan_discord_reconcile(
        [
            DiscordSourceMessage("1", "1", "ops", "Marci", kept[0], sent),
            DiscordSourceMessage("1", "2", "ops", "Marci", kept[1], sent),
            DiscordSourceMessage("1", "3", "ops", "Deploy Bot", kept[2], sent),
            DiscordSourceMessage("1", "4", "ops", "CI", "New documentation published", sent),
            DiscordSourceMessage("1", "5", "ops", "x-ci", "Ship portal fix today.", sent),
        ]
    )
    assert plan.deleted_ids == ()
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
    assert "not established as the requested document" in response.answer.casefold()
    assert evidence in response.answer
    assert secret not in response.answer
    assert _HOME not in response.answer


def _cited(chunk: SearchChunk, quote: str) -> AnswerResponse:
    return AnswerResponse(
        status="answered",
        answer="placeholder",
        confidence=0.9,
        citations=[
            Citation(
                source_file=chunk.source_file,
                chunk_id=chunk.id,
                content_sha256=_DIGEST,
                quote=quote,
            )
        ],
    )


@pytest.mark.unit
def test_plans_and_proposals_are_not_reported_as_completed_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planning = (
        "Kiosk next work is MVP definition and setting the timeline. "
        "Prototype todos remain open. "
        "The revised project proposal covers the architecture."
    )
    done = "BMTech approved the launch and the decision is final."
    plan_chunk = SearchChunk(205, "meeting.txt", 0, planning, {"source_type": "transcript", "kind": "meeting_transcript", "meeting": "sample", "title": "Planning", "url": "https://example.test/plan", "recording_id": "rec"}, 1.0, content_sha256=_DIGEST)
    done_chunk = SearchChunk(206, "meeting.txt", 1, done, {"source_type": "transcript", "kind": "meeting_transcript", "meeting": "sample", "title": "Decision", "url": "https://example.test/done", "recording_id": "rec"}, 1.0, content_sha256="c" * 64)

    def answer_for(question: str, chunk: SearchChunk, claim: str) -> AnswerResponse:
        monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
        monkeypatch.setattr(agent, "classify_query", lambda _question: "cheap")
        monkeypatch.setattr(
            agent,
            "retrieve_evidence",
            lambda *_a, **_k: RetrievalDecision(
                "accepted",
                (HybridSearchResult(chunk, 0.2, vector_score=0.9, fts_score=1.0),),
                0.9,
                1,
            ),
        )
        monkeypatch.setattr(
            agent,
            "call_llm",
            lambda **_k: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=""))]),
        )
        monkeypatch.setattr(
            agent,
            "call_llm_structured",
            lambda **_k: _cited(chunk, chunk.chunk_text).model_copy(update={"answer": claim}),
        )
        return agent.run_agent(question)

    qualified = answer_for(
        "What kiosk decisions are complete?",
        plan_chunk,
        "The MVP and timeline were defined, and the proposal architecture was decided.",
    )
    assert "does not explicitly state" in qualified.answer.casefold()
    assert "were defined" not in qualified.answer.casefold()
    assert "was decided" not in qualified.answer.casefold()
    assert "setting the timeline" in qualified.answer

    kept = answer_for(
        "What launch decision was approved?",
        done_chunk,
        "The launch was approved.",
    )
    assert kept.answer == "The launch was approved."


@pytest.mark.unit
def test_closest_page_is_not_renamed_as_the_requested_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    article = SearchChunk(
        110,
        "portal/guide/lifecycle.md",
        0,
        "This article explains a forward-deployed lifecycle. It is not a runbook.",
        {
            "source_type": "kb_page",
            "title": "Forward-deployed lifecycle",
            "url": "https://example.test/guide/lifecycle/",
        },
        1.0,
        content_sha256=_DIGEST,
    )
    runbook = SearchChunk(
        615,
        "portal/runbooks/index.md",
        0,
        "These runbooks and playbooks are the operating set.",
        {
            "source_type": "kb_page",
            "title": "Playbooks and runbooks",
            "url": "https://example.test/runbooks/",
        },
        1.0,
        content_sha256="d" * 64,
    )

    def answer_for(question: str, chunk: SearchChunk, claim: str) -> AnswerResponse:
        monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
        monkeypatch.setattr(agent, "classify_query", lambda _question: "cheap")
        monkeypatch.setattr(
            agent,
            "retrieve_evidence",
            lambda *_a, **_k: RetrievalDecision(
                "accepted",
                (HybridSearchResult(chunk, 0.2, vector_score=0.9, fts_score=1.0),),
                0.9,
                1,
            ),
        )
        monkeypatch.setattr(
            agent,
            "call_llm",
            lambda **_k: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=""))]),
        )
        monkeypatch.setattr(
            agent,
            "call_llm_structured",
            lambda **_k: _cited(chunk, chunk.chunk_text).model_copy(update={"answer": claim}),
        )
        return agent.run_agent(question)

    guide = answer_for(
        "Where is the deployment guide?",
        article,
        "The deployment guide is the forward-deployed lifecycle article.",
    )
    assert "not established as the requested document" in guide.answer.casefold()
    assert "closest cited page" in guide.answer.casefold()
    assert "forward-deployed lifecycle" in guide.answer.casefold()
    assert "the deployment guide is the forward-deployed" not in guide.answer.casefold()

    books = answer_for(
        "Where are the playbooks and runbooks?",
        runbook,
        "The playbooks and runbooks are the operating set.",
    )
    assert books.answer == "The playbooks and runbooks are the operating set."


@pytest.mark.unit
def test_discord_shaped_tokens_are_redacted_before_the_model() -> None:
    discord_token = "M" + ("a" * 23) + "." + ("b" * 6) + "." + ("c" * 27)
    mfa = "mfa." + ("d" * 40)
    redacted = redact_secrets(f"bot {discord_token} and {mfa} and version v1.2.3")
    assert discord_token not in redacted
    assert mfa not in redacted
    assert "v1.2.3" in redacted
    assert redacted.count("[REDACTED]") == 2


@pytest.mark.unit
def test_orphan_citation_marker_is_removed_and_unrelated_hire_abstains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planning = "Kiosk next work is MVP definition and setting the timeline. The revised project proposal covers architecture."
    prototype = "A demonstrated prototype exists only as a note in this planning paragraph."
    plan_chunk = SearchChunk(
        110,
        "notes/planning.md",
        0,
        planning,
        {"source_type": "kb_page", "title": "Planning note", "url": "https://example.test/planning/"},
        1.0,
        content_sha256=_DIGEST,
    )
    other = SearchChunk(
        200,
        "notes/other.md",
        0,
        prototype,
        {"source_type": "kb_page", "title": "Prototype note", "url": "https://example.test/prototype/"},
        1.0,
        content_sha256="e" * 64,
    )

    def run(question: str, chunks: tuple[SearchChunk, ...], claim: str, quotes: list[str]) -> AnswerResponse:
        monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
        monkeypatch.setattr(agent, "classify_query", lambda _question: "cheap")
        monkeypatch.setattr(
            agent,
            "retrieve_evidence",
            lambda *_a, **_k: RetrievalDecision(
                "accepted",
                tuple(HybridSearchResult(chunk, 0.2, vector_score=0.8, fts_score=1.0) for chunk in chunks),
                0.8,
                1,
            ),
        )
        monkeypatch.setattr(
            agent,
            "call_llm",
            lambda **_k: SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=""))]
            ),
        )

        def structured(**_kwargs: object) -> AnswerResponse:
            return AnswerResponse(
                status="answered",
                answer=claim,
                confidence=0.9,
                citations=[
                    Citation(
                        source_file=chunk.source_file,
                        chunk_id=chunk.id,
                        content_sha256=chunk.content_sha256 or _DIGEST,
                        quote=quote,
                    )
                    for chunk, quote in zip(chunks, quotes, strict=True)
                ],
            )

        monkeypatch.setattr(agent, "call_llm_structured", structured)
        return agent.run_agent(question)

    kiosk = run(
        "What kiosk decisions are complete?",
        (plan_chunk, other),
        "Architecture decisions were made for the kiosk [1]. The demonstrated prototype shipped [3].",
        [planning, prototype],
    )
    assert "does not explicitly state" in kiosk.answer.casefold()
    assert "[3]" not in kiosk.answer
    assert "demonstrated prototype shipped" not in kiosk.answer.casefold()
    assert [citation.chunk_id for citation in kiosk.citations] == [110, 200]

    decided = SearchChunk(
        300,
        "notes/decision.md",
        0,
        "The group decided to keep the prototype.",
        {"source_type": "kb_page", "title": "Decision note"},
        1.0,
        content_sha256="f" * 64,
    )
    kept = run(
        "What did the group decide about the prototype?",
        (decided,),
        "The group decided to keep the prototype.",
        ["The group decided to keep the prototype."],
    )
    assert kept.answer == "The group decided to keep the prototype."

    hire = run(
        "Who should we hire next for the core team?",
        (plan_chunk,),
        "Hire Nick to work on ingestion.",
        [planning],
    )
    assert hire.status == "no_relevant_information"
    assert hire.citations == []
    assert "Nick" not in hire.answer
    assert "does not address" in hire.answer.casefold()

    schedule = SearchChunk(
        521,
        "notes/schedule.md",
        0,
        "A scheduler can be triggered manually. Suggest a bot if you want automation.",
        {"source_type": "kb_page", "title": "Scheduler"},
        1.0,
        content_sha256="a" * 64,
    )
    howto = run(
        "How do I index a new Discord channel?",
        (schedule,),
        "Trigger the ingestion workflow and add a bot.",
        ["A scheduler can be triggered manually. Suggest a bot if you want automation."],
    )
    assert howto.status == "no_relevant_information"
    assert "ingestion workflow" not in howto.answer.casefold()
    assert "unknown" in howto.answer.casefold()
    assert howto.citations == []

    lifecycle = SearchChunk(
        110,
        "portal/guide/lifecycle.md",
        0,
        "Later we might write a deployment guide. This article explains a lifecycle.",
        {
            "source_type": "kb_page",
            "title": "Forward-deployed lifecycle",
            "url": "https://example.test/guide/lifecycle/",
        },
        1.0,
        content_sha256=_DIGEST,
    )
    guide = run(
        "Where is the deployment guide?",
        (lifecycle,),
        "The deployment guide is the forward-deployed lifecycle article.",
        ["Later we might write a deployment guide. This article explains a lifecycle."],
    )
    assert "not established as the requested document" in guide.answer.casefold()
    assert "the deployment guide is the forward-deployed" not in guide.answer.casefold()


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


@pytest.mark.unit
def test_full_name_is_a_roster_and_a_place_is_not(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "call_llm", lambda **_kwargs: pytest.fail("roster extractor must run first"))
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_a, **_k: RetrievalDecision(
            "accepted",
            (_result("Ada Lovelace is on the core team."),),
            0.8,
            1,
        ),
    )

    response = agent.run_agent("Who is on the core team?")

    assert response.status == "answered"
    assert "Ada Lovelace" in response.answer
    assert "Boston" not in response.answer


@pytest.mark.unit
def test_entailment_keeps_supported_text_and_drops_invented_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    certificate = "Rotate the certificate in the portal settings."
    planning = (
        "Kiosk next work is MVP definition and setting the timeline. "
        "The revised project proposal covers architecture."
    )
    hiring = "The core team will discuss hiring next month."
    comparison = "A playbook explains the approach. A runbook lists the steps."

    def run(question: str, chunk: SearchChunk, claim: str, quote: str | None = None) -> AnswerResponse:
        monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
        monkeypatch.setattr(agent, "classify_query", lambda _question: "cheap")
        monkeypatch.setattr(
            agent,
            "retrieve_evidence",
            lambda *_a, **_k: RetrievalDecision(
                "accepted",
                (HybridSearchResult(chunk, 0.2, vector_score=0.8, fts_score=1.0),),
                0.8,
                1,
            ),
        )
        monkeypatch.setattr(
            agent,
            "call_llm",
            lambda **_k: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=""))]),
        )
        monkeypatch.setattr(
            agent,
            "call_llm_structured",
            lambda **_k: AnswerResponse(
                status="answered",
                answer=claim,
                confidence=0.9,
                citations=[
                    Citation(
                        source_file=chunk.source_file,
                        chunk_id=chunk.id,
                        content_sha256=chunk.content_sha256 or _DIGEST,
                        quote=quote or chunk.chunk_text,
                    )
                ],
            ),
        )
        return agent.run_agent(question)

    cert_chunk = SearchChunk(
        8,
        "portal/security.md",
        0,
        certificate,
        {"source_type": "kb_page", "title": "Security"},
        1.0,
        content_sha256=_DIGEST,
    )
    rotated = run("How do I rotate the portal certificate?", cert_chunk, certificate)
    assert rotated.status == "answered"
    assert certificate in rotated.answer
    assert "no_relevant_information" not in rotated.answer

    cafeteria = SearchChunk(
        9,
        "notes/office.md",
        0,
        "Open the cafeteria.",
        {"source_type": "kb_page", "title": "Office"},
        1.0,
        content_sha256=_DIGEST,
    )
    opened = run("How do I open the portal?", cafeteria, "Open the cafeteria.")
    assert opened.status == "no_relevant_information"
    assert "supported steps" not in opened.answer.casefold()
    assert "cafeteria" not in opened.answer.casefold()
    assert "no_relevant_information" not in opened.answer

    plan_chunk = SearchChunk(
        110,
        "notes/planning.md",
        0,
        planning,
        {"source_type": "kb_page", "title": "Planning note"},
        1.0,
        content_sha256=_DIGEST,
    )
    made = run(
        "What kiosk decisions are complete?",
        plan_chunk,
        "Architecture decisions made for the kiosk.",
    )
    assert made.status == "answered"
    assert "does not explicitly state" in made.answer.casefold()
    assert "not a final decision" in made.answer.casefold()
    assert "setting the timeline" in made.answer
    assert "decisions made" not in made.answer.casefold()

    hire_chunk = SearchChunk(
        12,
        "notes/hiring.md",
        0,
        hiring,
        {"source_type": "kb_page", "title": "Hiring note"},
        1.0,
        content_sha256="c" * 64,
    )
    hired = run("Who did the core team hire?", hire_chunk, "The core team hired Ada.")
    assert "ada" not in hired.answer.casefold()
    assert "hired ada" not in hired.answer.casefold()
    assert "discuss hiring" in hired.answer.casefold()
    assert "does not explicitly state" in hired.answer.casefold()

    guide = SearchChunk(
        615,
        "portal/guide/index.md",
        0,
        comparison,
        {"source_type": "kb_page", "title": "Guide", "url": "https://example.test/guide/"},
        1.0,
        content_sha256="d" * 64,
    )
    compared = run("How do playbooks compare with runbooks?", guide, comparison)
    assert compared.answer == comparison
    assert "identified as the runbook" not in compared.answer.casefold()
    assert "closest cited page" not in compared.answer.casefold()

    named = SearchChunk(
        70,
        "kb/deployment-guide.md",
        0,
        "Use this deployment guide for releases.",
        {"source_type": "kb_page", "title": "", "url": ""},
        1.0,
        content_sha256="e" * 64,
    )
    located = run(
        "Where is the deployment guide?",
        named,
        "The deployment guide is this page.",
    )
    assert "not established as the requested document" not in located.answer.casefold()
    assert "deployment guide" in located.answer.casefold()

    mixed = run(
        "How do I rotate the portal certificate?",
        cert_chunk,
        f"The portal certificate expired. {certificate} [1].",
    )
    assert "expired" not in mixed.answer.casefold()
    assert certificate in mixed.answer
    assert mixed.citations[0].chunk_id == 8


@pytest.mark.unit
def test_abstention_uses_plain_language_and_keeps_a_cited_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planning = "Kiosk next work is MVP definition and setting the timeline."
    chunk = SearchChunk(
        110,
        "notes/planning.md",
        0,
        planning,
        {"source_type": "kb_page", "title": "Planning note"},
        1.0,
        content_sha256=_DIGEST,
    )

    def stub(question_chunk: SearchChunk, response: AnswerResponse) -> None:
        monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
        monkeypatch.setattr(agent, "classify_query", lambda _question: "cheap")
        monkeypatch.setattr(
            agent,
            "retrieve_evidence",
            lambda *_a, **_k: RetrievalDecision(
                "accepted",
                (HybridSearchResult(question_chunk, 0.2, vector_score=0.8, fts_score=1.0),),
                0.8,
                1,
            ),
        )
        monkeypatch.setattr(
            agent,
            "call_llm",
            lambda **_k: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=""))]),
        )
        monkeypatch.setattr(agent, "call_llm_structured", lambda **_k: response)

    stub(
        chunk,
        AnswerResponse(status="no_relevant_information", answer="no_relevant_information", confidence=0.0),
    )
    kiosk = agent.run_agent("What kiosk decisions are complete?")
    assert kiosk.status == "answered"
    assert "setting the timeline" in kiosk.answer
    assert "does not explicitly state" in kiosk.answer.casefold()
    assert "not a final decision" in kiosk.answer.casefold()
    assert "no_relevant_information" not in kiosk.answer
    assert kiosk.citations[0].chunk_id == 110

    billing = SearchChunk(
        4,
        "portal/guide/billing.md",
        0,
        "Billing settings live on another team.",
        {"source_type": "kb_page", "title": "Billing"},
        1.0,
        content_sha256=_DIGEST,
    )
    stub(
        billing,
        AnswerResponse(status="no_relevant_information", answer="no_relevant_information", confidence=0.0),
    )
    unknown = agent.run_agent("What's the URL for the billing page in our KB?")
    assert unknown.status == "no_relevant_information"
    assert unknown.citations == []
    assert "no_relevant_information" not in unknown.answer
    assert "does not address" in unknown.answer.casefold()
    assert "incident" not in unknown.answer.casefold()


@pytest.mark.unit
def test_claim_entailment_accepts_paraphrase_and_marks_status() -> None:
    assert (
        assess_claim(
            "The page describes invoices and credits.",
            "The billing page covers invoices and credits.",
        )
        == "supported"
    )
    assert (
        assess_claim(
            "It describes invoices and credits.",
            "The billing page covers invoices and credits.",
        )
        == "supported"
    )
    assert (
        assess_claim(
            "The playbook says to page the primary and then post in incident channel.",
            "Page the primary, then post in incident channel.",
        )
        == "supported"
    )
    billed = "Customers are billed through invoices and credits."
    assert assess_claim(billed, billed) == "supported"
    weaker = assess_claim(billed, "The page covers invoices and credits.")
    assert weaker == "unknown"
    assert (
        assess_claim(
            "The page describes invoices and credits.",
            "The guide does not cover invoices or credits.",
        )
        == "contradicted"
    )
    assert (
        assess_claim(
            "The core team hired Ada.",
            "The core team will discuss hiring next month.",
        )
        == "unknown"
    )


@pytest.mark.unit
def test_ordinary_answers_stay_meaningful_and_responsive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(question: str, chunk: SearchChunk, claim: str, quote: str | None = None) -> AnswerResponse:
        monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
        monkeypatch.setattr(agent, "classify_query", lambda _question: "cheap")
        monkeypatch.setattr(
            agent,
            "retrieve_evidence",
            lambda *_a, **_k: RetrievalDecision(
                "accepted",
                (HybridSearchResult(chunk, 0.2, vector_score=0.8, fts_score=1.0),),
                0.8,
                1,
            ),
        )
        monkeypatch.setattr(
            agent,
            "call_llm",
            lambda **_k: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=""))]),
        )
        monkeypatch.setattr(
            agent,
            "call_llm_structured",
            lambda **_k: AnswerResponse(
                status="answered",
                answer=claim,
                confidence=0.9,
                citations=[
                    Citation(
                        source_file=chunk.source_file,
                        chunk_id=chunk.id,
                        content_sha256=chunk.content_sha256 or _DIGEST,
                        quote=quote or chunk.chunk_text,
                    )
                ],
            ),
        )
        return agent.run_agent(question)

    comparison = "A playbook explains the approach. A runbook lists the steps."
    guide = SearchChunk(
        7,
        "portal/guide/operations.md",
        0,
        comparison,
        {"source_type": "kb_page", "title": "Guide", "url": "https://example.test/guide/"},
        1.0,
        content_sha256=_DIGEST,
    )
    paraphrase = "Playbooks explain the approach, while runbooks list the steps."
    compared = run("How do playbooks compare with runbooks?", guide, paraphrase)
    assert compared.status == "answered"
    assert compared.answer == paraphrase
    assert "incident" not in compared.answer.casefold()
    assert "playbook" in compared.answer.casefold()
    assert "runbook" in compared.answer.casefold()

    suppressed = run(
        "How do playbooks compare with runbooks?",
        guide,
        "The retrieved evidence does not address this question. Missing facts, including any current incident state, are unknown.",
    )
    assert suppressed.status == "answered"
    assert "playbook" in suppressed.answer.casefold()
    assert "runbook" in suppressed.answer.casefold()
    assert "current incident" not in suppressed.answer.casefold()

    numbered = SearchChunk(
        8,
        "notes/planning.md",
        0,
        "Kiosk next work is MVP definition and setting the timeline.",
        {"source_type": "kb_page", "title": "Planning note"},
        1.0,
        content_sha256="c" * 64,
    )
    kiosk = run("What kiosk decisions are complete?", numbered, "2. 3. 4.")
    assert meaningful_text(kiosk.answer)
    assert re.fullmatch(r"[\d.\s]+", kiosk.answer.strip()) is None
    assert "2. 3. 4." != kiosk.answer.strip()
    assert "timeline" in kiosk.answer.casefold()
    assert "does not explicitly state" in kiosk.answer.casefold()

    link = SearchChunk(
        9,
        "notes/chat.md",
        0,
        "Can someone share the Google link?",
        {"source_type": "discord", "title": "Chat"},
        1.0,
        content_sha256="d" * 64,
    )
    oncall = run("How many staff are on the on-call rotation?", link, "Can someone share the Google link?")
    assert oncall.status == "no_relevant_information"
    assert oncall.citations == []
    assert "google" not in oncall.answer.casefold()
    assert "does not address" in oncall.answer.casefold()

    brainstorm = SearchChunk(
        10,
        "notes/ideas.md",
        0,
        "Brainstorm note: we might hire someone for ingestion someday.",
        {"source_type": "kb_page", "title": "Old ideas"},
        1.0,
        content_sha256="e" * 64,
    )
    hiring = run(
        "Who should we hire next for the core team?",
        brainstorm,
        "Hire Nick to own ingestion.",
    )
    assert "nick" not in hiring.answer.casefold()
    assert "not a current recommendation" in hiring.answer.casefold()
    assert "brainstorm" in hiring.answer.casefold()

    billed = "Customers are billed through invoices and credits."
    billing = SearchChunk(
        11,
        "portal/billing.md",
        0,
        billed,
        {"source_type": "kb_page", "title": "Billing"},
        1.0,
        content_sha256="f" * 64,
    )
    invoices = run("What does billing cover?", billing, billed)
    assert invoices.status == "answered"
    assert invoices.answer == billed
    assert "not a final decision" not in invoices.answer.casefold()
    covers = SearchChunk(
        13,
        "portal/billing.md",
        0,
        "The page covers invoices and credits.",
        {"source_type": "kb_page", "title": "Billing"},
        1.0,
        content_sha256="b" * 64,
    )
    described = run("What does the page describe?", covers, "The page describes invoices and credits.")
    assert described.status == "answered"
    assert "invoices and credits" in described.answer.casefold()
    assert "not a final decision" not in described.answer.casefold()
    hedged = run("What does the page describe?", covers, billed)
    assert "not a final decision" not in hedged.answer.casefold()
    steps = supported_steps(
        "Page the primary, then post in incident channel, then call the vendor.",
        "Page the primary, then post in incident channel.",
    )
    assert steps is not None
    assert "vendor" not in steps.casefold()
    assert "primary" in steps.casefold()

    duties = SearchChunk(
        12,
        "portal/guide/lifecycle.md",
        0,
        "Forward-deployed engineers own release responsibilities.",
        {"source_type": "kb_page", "title": "Forward-deployed responsibilities"},
        1.0,
        content_sha256="a" * 64,
    )
    located = run(
        "Where is the deployment guide?",
        duties,
        "Forward-deployed engineers own release responsibilities.",
    )
    assert "closest cited page" in located.answer.casefold()
    assert located.answer.strip() != duties.chunk_text

    dumped = scrub_response(
        AnswerResponse(status="no_relevant_information", answer="status : nothing matched", confidence=0.0)
    )
    assert "nothing matched" not in dumped.answer.casefold()
    assert "status" not in dumped.answer.casefold()
    assert "does not address" in dumped.answer.casefold()
