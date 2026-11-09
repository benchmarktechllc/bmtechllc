"""Non-posting acceptance run for Ask BMTech.

This calls ``run_agent``, the same function the Discord bot uses. It does not
connect to Discord and does not send messages. Retrieval embeds each question,
so ``DATABASE_URL`` and ``OPENAI_API_KEY`` must already be in the environment.
``PORTAL_BASE_URL`` must be the current GitHub Pages portal. The report lists
configured/missing dependencies without printing secret values.
"""

from __future__ import annotations

import json
import os
import re

from .agent import run_agent
from .corpus import is_boilerplate_text, is_synthetic_canary
from .grounding import (
    ACCEPTANCE_QUESTIONS,
    CHANNEL_ACCEPTANCE_QUESTIONS,
    CORE_TEAM_ACCEPTANCE_QUESTIONS,
    EMPLOYEE_ACCEPTANCE_QUESTIONS,
    PORTAL_ACCEPTANCE_QUESTIONS,
    asks_for_core_team,
    asks_for_employee_count,
    asks_for_portal_link,
    asks_for_synced_channels,
)
from .portal_urls import portal_home_url

_DIGIT_EMPLOYEE_COUNT = re.compile(r"\b\d+\s+employees?\b", re.IGNORECASE)
_STALE_HOST = "hussein881.github.io"
_GENERIC_NO_INFO = "no relevant information was found in the company document library"
_COUNT_UNCERTAINTY = "does not state a current employee count"
_NO_CORE_TEAM = "does not name a core team"
_CHANNEL_PROVENANCE = "Indexed Discord channel #"

# Public repo search found no page that names a BMTech core-team roster.
# A reviewer handle in another project is not that roster.
CORE_TEAM_EXPECTATION = (
    "unknown_ok. No explicit core-team roster is in the public repo. "
    "Abstaining with no citations is a data gap, not proof that a roster was retrieved."
)
EMPLOYEE_EXPECTATION = (
    "unknown_ok unless an indexed sentence states a digit employee count. "
    "Uncertainty with no citations is the safe result, not a measured headcount."
)


def _dependencies() -> dict[str, bool]:
    return {
        "database_url_configured": bool(os.getenv("DATABASE_URL", "").strip()),
        "openai_api_key_configured": bool(os.getenv("OPENAI_API_KEY", "").strip())
        and os.getenv("OPENAI_API_KEY", "").strip() != "your_openai_api_key_here",
        "portal_base_configured": portal_home_url() is not None,
    }


def _displayed_quotes(response_payload: dict[str, object]) -> list[str]:
    citations = response_payload.get("citations")
    if not isinstance(citations, list):
        return []
    return [str(citation.get("quote") or "") for citation in citations if isinstance(citation, dict)]


def _citation_views(response_payload: dict[str, object]) -> list[dict[str, object]]:
    """Citation identifiers only. Message text stays out of the report."""
    citations = response_payload.get("citations")
    if not isinstance(citations, list):
        return []
    views: list[dict[str, object]] = []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        source = str(citation.get("source_file") or "")
        quote = str(citation.get("quote") or "")
        views.append(
            {
                "source_file": source,
                "chunk_id": citation.get("chunk_id"),
                "synthetic_source": is_synthetic_canary(source, {}),
                "boilerplate_quote": is_boilerplate_text(quote),
                "contains_previous_pages_host": _STALE_HOST in quote,
                "contains_url": "http://" in quote or "https://" in quote,
                "provenance_only": quote.startswith(_CHANNEL_PROVENANCE),
            }
        )
    return views


def _check(question: str, payload: dict[str, object], *, home: str | None) -> list[str]:
    answer = str(payload.get("answer") or "")
    status = str(payload.get("status") or "")
    quotes = _displayed_quotes(payload)
    displayed = "\n".join([answer, *quotes])
    failures: list[str] = []
    if _STALE_HOST in displayed:
        failures.append("displayed answer or citation contains the previous Pages host")
    views = _citation_views(payload)
    if any(view["synthetic_source"] for view in views):
        failures.append("citation source is a synthetic canary")
    if any(view["boilerplate_quote"] for view in views) or is_boilerplate_text(answer):
        failures.append("answer or citation is indexed bot boilerplate")
    if _GENERIC_NO_INFO in answer.casefold():
        failures.append("generic no-information sentence")
    if asks_for_portal_link(question):
        if home is None:
            failures.append("PORTAL_BASE_URL is not set")
        elif home not in answer:
            failures.append(f"portal answer does not contain {home}")
        elif status != "answered":
            failures.append("portal question was not answered")
        elif not views:
            failures.append("portal answer has no citation")
    elif asks_for_employee_count(question):
        if _DIGIT_EMPLOYEE_COUNT.search(answer):
            failures.append("employee answer states a numeric headcount")
        if _COUNT_UNCERTAINTY not in answer.casefold():
            failures.append("employee answer does not state count uncertainty")
    elif asks_for_core_team(question):
        if _DIGIT_EMPLOYEE_COUNT.search(answer):
            failures.append("core-team answer states a numeric headcount")
        if status != "answered" and _NO_CORE_TEAM not in answer.casefold():
            failures.append("core-team data gap is not explained")
        if status == "answered" and not views:
            failures.append("core-team answer has no citation")
    elif asks_for_synced_channels(question):
        if status != "answered" or "#" not in answer:
            failures.append("synced-channel answer did not list indexed channel names")
        elif not views or not all(view["provenance_only"] for view in views):
            failures.append("channel citation is not limited to channel provenance")
        elif any(view["contains_url"] for view in views):
            failures.append("channel citation includes a URL from message text")
    return failures


def _outcome(question: str, payload: dict[str, object], failures: list[str]) -> dict[str, object]:
    """Proof requires the answer's claim, not merely status == answered."""
    status = str(payload.get("status") or "")
    answer = str(payload.get("answer") or "")
    citations = payload.get("citations")
    has_citations = isinstance(citations, list) and bool(citations)
    if failures:
        return {"outcome": "failed", "proves_fact": False, "expectation": "must_answer"}
    if asks_for_portal_link(question):
        proves = status == "answered" and "https://" in answer and has_citations
        return {"outcome": "answered" if proves else "unknown", "proves_fact": proves, "expectation": "must_answer"}
    if asks_for_synced_channels(question):
        proves = status == "answered" and "#" in answer and has_citations
        return {"outcome": "answered" if proves else "unknown", "proves_fact": proves, "expectation": "must_answer"}
    if asks_for_employee_count(question):
        proves = (
            status == "answered"
            and _DIGIT_EMPLOYEE_COUNT.search(answer) is not None
            and _COUNT_UNCERTAINTY not in answer.casefold()
            and has_citations
        )
        return {
            "outcome": "answered" if proves else "unknown",
            "proves_fact": proves,
            "expectation": "unknown_ok",
        }
    if asks_for_core_team(question):
        proves = (
            status == "answered"
            and _NO_CORE_TEAM not in answer.casefold()
            and "not an employee" in answer.casefold()
            and has_citations
        )
        return {
            "outcome": "answered" if proves else "unknown",
            "proves_fact": proves,
            "expectation": "unknown_ok",
        }
    return {"outcome": "unknown", "proves_fact": False, "expectation": "not_classified"}


def run_acceptance() -> dict[str, object]:
    """Answer the production questions and paraphrases, then report grounded checks."""
    dependencies = _dependencies()
    home = portal_home_url()
    failures: list[str] = []
    if not dependencies["database_url_configured"]:
        failures.append("DATABASE_URL is not configured")
    if not dependencies["openai_api_key_configured"]:
        failures.append("OPENAI_API_KEY is not configured")
    results = []
    if dependencies["database_url_configured"] and dependencies["openai_api_key_configured"]:
        for question in ACCEPTANCE_QUESTIONS:
            response = run_agent(question)
            payload = response.model_dump()
            question_failures = _check(question, payload, home=home)
            for item in question_failures:
                failures.append(f"{question}: {item}")
            results.append(
                {
                    "question": question,
                    "status": payload.get("status"),
                    "answer": payload.get("answer"),
                    "citations": _citation_views(payload),
                    "failures": question_failures,
                    **_outcome(question, payload, question_failures),
                }
            )
    return {
        "revision": "2026-11-07-ask-retrieval",
        "portal_home": home,
        "dependencies": dependencies,
        "expectations": {
            "portal": "must_answer with the configured absolute portal URL and a knowledge-base citation.",
            "channels": (
                "must_answer with indexed channel names. Citations are provenance labels. "
                "A historic URL may remain in the stored chunk and is not rewritten or displayed."
            ),
            "employee_count": EMPLOYEE_EXPECTATION,
            "core_team": CORE_TEAM_EXPECTATION,
        },
        "question_groups": {
            "portal": list(PORTAL_ACCEPTANCE_QUESTIONS),
            "employee": list(EMPLOYEE_ACCEPTANCE_QUESTIONS),
            "core_team": list(CORE_TEAM_ACCEPTANCE_QUESTIONS),
            "channels": list(CHANNEL_ACCEPTANCE_QUESTIONS),
        },
        "failures": failures,
        "results": results,
    }


def main() -> None:
    report = run_acceptance()
    print(json.dumps(report, indent=2))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
