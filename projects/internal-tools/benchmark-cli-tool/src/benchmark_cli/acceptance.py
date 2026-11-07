"""Non-posting acceptance run for the four Ask BMTech questions.

This calls ``run_agent``, the same function the Discord bot uses. It does not
connect to Discord and does not send messages. It needs ``DATABASE_URL`` and
``OPENAI_API_KEY`` because retrieval embeds the question. Set
``PORTAL_BASE_URL`` to the current GitHub Pages portal before expecting a
clickable link.
"""

from __future__ import annotations

import json
import re

from .agent import run_agent
from .grounding import ACCEPTANCE_QUESTIONS
from .portal_urls import portal_home_url

_EMPLOYEE_COUNT = re.compile(r"\b\d+\s+employees?\b", re.IGNORECASE)
_STALE_HOST = "hussein881.github.io"
_ARTICLE_BOT = "new documentation published"


def run_acceptance() -> dict[str, object]:
    """Answer the four production questions and report grounded checks."""
    home = portal_home_url()
    results = []
    failures: list[str] = []
    for question in ACCEPTANCE_QUESTIONS:
        response = run_agent(question)
        payload = response.model_dump()
        blob = json.dumps(payload)
        if _STALE_HOST in blob:
            failures.append(f"{question}: answer still contains the previous Pages host")
        if question.startswith("What is the link"):
            if home is None:
                failures.append("PORTAL_BASE_URL is not set, so no canonical portal URL is available")
            elif home not in response.answer:
                failures.append(f"portal answer does not contain {home}")
            elif response.status != "answered":
                failures.append("portal question was not answered")
        elif question.startswith("How many employees"):
            if _EMPLOYEE_COUNT.search(response.answer):
                failures.append("employee answer states a numeric headcount")
            if "employee count" not in response.answer.casefold():
                failures.append("employee answer does not say whether a count is in evidence")
        elif question.startswith("Who are the core team"):
            if _EMPLOYEE_COUNT.search(response.answer):
                failures.append("core-team answer states a numeric headcount")
        elif question.startswith("Which Discord"):
            if response.status != "answered" or "#" not in response.answer:
                failures.append("synced-channel answer did not list indexed channel names")
            if _ARTICLE_BOT in blob.casefold():
                failures.append("synced-channel answer used the article-update notification")
        results.append({"question": question, "response": payload})
    return {"portal_home": home, "failures": failures, "results": results}


def main() -> None:
    report = run_acceptance()
    print(json.dumps(report, indent=2))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
