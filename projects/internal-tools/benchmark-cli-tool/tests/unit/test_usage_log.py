"""Usage logs must be writable by the service user and must not fail an answer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from benchmark_cli.logging import log_agent_diagnostic, log_usage, usage_log_path


@pytest.mark.unit
def test_log_dir_env_is_preferred_over_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    xdg = tmp_path / "xdg"
    override = tmp_path / "override"
    monkeypatch.setenv("XDG_STATE_HOME", str(xdg))
    monkeypatch.setenv("BENCHMARK_LOG_DIR", str(override))

    log_usage(tier="cheap", prompt_tokens=3, completion_tokens=1, latency_ms=1.5, component="agent")

    written = override / "model_usage.csv"
    assert written.is_file()
    assert "site-packages" not in written.parts
    assert not (xdg / "benchmark-cli-tool" / "model_usage.csv").exists()
    text = written.read_text(encoding="utf-8")
    assert "agent" in text
    assert "cheap" in text


@pytest.mark.unit
def test_xdg_state_is_used_when_no_env_override_is_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("BENCHMARK_LOG_DIR", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    log_agent_diagnostic(outcome="answered", synthesis_attempts=1)

    written = tmp_path / "benchmark-cli-tool" / "agent_diagnostics.csv"
    assert written.is_file()
    assert "answered" in written.read_text(encoding="utf-8")


@pytest.mark.unit
def test_temp_dir_is_used_when_env_and_home_are_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BENCHMARK_LOG_DIR", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("HOME", raising=False)

    path = usage_log_path()

    assert path == Path(tempfile.gettempdir()) / "benchmark-cli-tool" / "model_usage.csv"
    assert "site-packages" not in path.parts


@pytest.mark.unit
def test_permission_error_does_not_fail_the_caller(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    blocked = tmp_path / "site-packages" / "model_usage.csv"
    real_open = Path.open

    def guarded(self: Path, mode: str = "r", *args: object, **kwargs: object) -> object:
        if self == blocked:
            raise PermissionError("read-only site-packages")
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)

    log_usage(
        tier="cheap",
        prompt_tokens=1,
        completion_tokens=1,
        latency_ms=1,
        component="agent",
        log_path=blocked,
    )
    log_agent_diagnostic(outcome="answered", synthesis_attempts=0, log_path=blocked)

    assert not blocked.exists()
