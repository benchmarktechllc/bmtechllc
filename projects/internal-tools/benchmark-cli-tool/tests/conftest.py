"""Keep unit tests offline. Query understanding is opted in by each test."""

from __future__ import annotations

import pytest

from benchmark_cli.query_understanding import fallback_plan


@pytest.fixture(autouse=True)
def offline_query_understanding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not call the cheap model unless a test invokes ``understand_query`` itself."""
    monkeypatch.setattr("benchmark_cli.agent.understand_query", fallback_plan)
