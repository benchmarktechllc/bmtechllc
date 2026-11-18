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


@pytest.mark.unit
def test_chat_tier_mapping() -> None:
    from benchmark_cli.llm import _chat_tier

    assert _chat_tier("synthesis") == "synthesis"
    assert _chat_tier("embedding") == "embedding"
    assert _chat_tier("flagship") == "cheap"
    assert _chat_tier("cheap") == "cheap"


@pytest.mark.unit
def test_structured_call_sends_the_tier_model_and_no_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from pydantic import BaseModel

    from benchmark_cli import llm
    from benchmark_cli.config import CHEAP_MODEL, get_model_config

    class Out(BaseModel):
        text: str = ""

    requests: list[dict[str, object]] = []

    def parse(**kwargs: object) -> object:
        requests.append(kwargs)
        message = SimpleNamespace(parsed=Out())
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    monkeypatch.setattr(llm, "log_usage", lambda **_k: None)
    monkeypatch.setattr(llm, "SYNTHESIS_REASONING_EFFORT", "")
    messages = [{"role": "user", "content": "q"}]
    llm.call_llm_structured(tier="synthesis", messages=messages, response_schema=Out)
    llm.call_llm_structured(tier="cheap", messages=messages, response_schema=Out)
    assert requests[0]["model"] == get_model_config("synthesis").model
    assert requests[1]["model"] == CHEAP_MODEL
    for request in requests:
        assert not {"max_tokens", "max_completion_tokens", "temperature", "reasoning_effort"} & set(request)


@pytest.mark.unit
def test_reasoning_effort_only_for_gpt5_models_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from pydantic import BaseModel

    from benchmark_cli import llm
    from benchmark_cli.config import ModelConfig

    class Out(BaseModel):
        text: str = ""

    requests: list[dict[str, object]] = []

    def parse(**kwargs: object) -> object:
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=Out()))], usage=None)

    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    monkeypatch.setattr(llm, "log_usage", lambda **_k: None)
    monkeypatch.setattr(llm, "SYNTHESIS_REASONING_EFFORT", "low")
    messages = [{"role": "user", "content": "q"}]
    llm.call_llm_structured(tier="synthesis", messages=messages, response_schema=Out)
    assert requests[-1].get("reasoning_effort") == "low"
    monkeypatch.setitem(llm.get_model_config.__globals__["MODEL_TIERS"], "synthesis", ModelConfig("gpt-4.1-mini", 0.4, 1.6))
    llm.call_llm_structured(tier="synthesis", messages=messages, response_schema=Out)
    assert "reasoning_effort" not in requests[-1]


@pytest.mark.unit
def test_synthesis_usage_is_logged_with_its_model_and_prices(tmp_path: Path) -> None:
    from benchmark_cli.config import get_model_config

    log_usage(
        tier="synthesis", prompt_tokens=1_000_000, completion_tokens=1_000_000, latency_ms=1.0,
        component="answer_synthesis", log_path=tmp_path / "usage.csv",
    )
    config = get_model_config("synthesis")
    text = (tmp_path / "usage.csv").read_text(encoding="utf-8")
    assert config.model in text and "synthesis" in text
    expected = config.input_cost_per_million + config.output_cost_per_million
    assert f"{expected:.10f}" in text
