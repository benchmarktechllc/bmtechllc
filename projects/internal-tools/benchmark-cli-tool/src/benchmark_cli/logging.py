"""Append-only usage telemetry with routed-versus-flagship cost accounting.

Logs are not written next to the installed package. A wheel lives under
site-packages, which the service user cannot modify, so a failed log used to
raise PermissionError before every answer. Writes go to BENCHMARK_LOG_DIR,
then an XDG state or cache directory, then the system temp directory, and a
failure is recorded without failing the answer.
"""

from __future__ import annotations

import csv
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from .config import ModelConfig, get_model_config
from .paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Legacy project path. Runtime writes use usage_log_path() instead, because
# this location is inside the install tree after `pip install`.
USAGE_LOG_PATH = PROJECT_ROOT / "artifacts" / "model_usage.csv"
AGENT_DIAGNOSTIC_LOG_PATH = PROJECT_ROOT / "artifacts" / "agent_diagnostics.csv"
CSV_FIELDS = (
    "timestamp_utc",
    "component",
    "tier",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "latency_ms",
    "actual_cost_usd",
    "flagship_equivalent_cost_usd",
    "routing_savings_usd",
)
AGENT_DIAGNOSTIC_FIELDS = ("timestamp_utc", "outcome", "synthesis_attempts")
_USAGE_FILENAME = "model_usage.csv"
_DIAGNOSTIC_FILENAME = "agent_diagnostics.csv"


def _candidate_directories() -> list[Path]:
    """Env override, then XDG, then temp. The install tree is not a candidate."""
    ordered: list[Path] = []
    override = os.getenv("BENCHMARK_LOG_DIR", "").strip()
    if override:
        ordered.append(Path(override).expanduser())
    xdg_state = os.getenv("XDG_STATE_HOME", "").strip()
    if xdg_state:
        ordered.append(Path(xdg_state).expanduser() / "benchmark-cli-tool")
    else:
        xdg_cache = os.getenv("XDG_CACHE_HOME", "").strip()
        if xdg_cache:
            ordered.append(Path(xdg_cache).expanduser() / "benchmark-cli-tool")
        home = os.getenv("HOME", "").strip()
        if home:
            ordered.append(Path(home).expanduser() / ".local" / "state" / "benchmark-cli-tool")
    ordered.append(Path(tempfile.gettempdir()) / "benchmark-cli-tool")
    unique: list[Path] = []
    seen: set[Path] = set()
    for directory in ordered:
        if directory in seen:
            continue
        seen.add(directory)
        unique.append(directory)
    return unique


def telemetry_directory() -> Path:
    """First writable telemetry directory, or the system temp directory."""
    for directory in _candidate_directories():
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / ".write-probe"
            probe.touch()
            probe.unlink(missing_ok=True)
        except OSError:
            continue
        else:
            return directory
    return Path(tempfile.gettempdir())


def usage_log_path() -> Path:
    """CSV path for model-call telemetry."""
    return telemetry_directory() / _USAGE_FILENAME


def agent_diagnostic_log_path() -> Path:
    """CSV path for answer-agent outcomes."""
    return telemetry_directory() / _DIAGNOSTIC_FILENAME


def _write_csv(path: Path, fields: tuple[str, ...], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _persist(explicit: Path | None, filename: str, fields: tuple[str, ...], row: dict[str, str]) -> None:
    """Append one row. Permission errors are logged and do not propagate."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    else:
        candidates.extend(directory / filename for directory in _candidate_directories())
    fallback = Path(tempfile.gettempdir()) / "benchmark-cli-tool" / filename
    if all(path != fallback for path in candidates):
        candidates.append(fallback)
    for path in candidates:
        try:
            _write_csv(path, fields, row)
        except OSError:
            logger.warning("Could not append telemetry to %s.", path, exc_info=True)
            continue
        return
    logger.warning("Telemetry for %s was skipped because no directory was writable.", filename)


def calculate_cost_usd(model: ModelConfig, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate USD cost from published per-million-token prices."""
    return (
        prompt_tokens * model.input_cost_per_million + completion_tokens * model.output_cost_per_million
    ) / 1_000_000


def log_usage(
    *,
    tier: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    component: str,
    log_path: Path | None = None,
) -> None:
    """Persist one model call without storing user or document content."""
    if prompt_tokens < 0 or completion_tokens < 0 or latency_ms < 0:
        raise ValueError("Token counts and latency must not be negative.")
    model = get_model_config(tier)
    actual = calculate_cost_usd(model, prompt_tokens, completion_tokens)
    flagship = calculate_cost_usd(
        get_model_config("flagship"), prompt_tokens, completion_tokens
    )
    equivalent = flagship if tier == "cheap" and component != "embedding" else actual
    _persist(
        log_path,
        _USAGE_FILENAME,
        CSV_FIELDS,
        {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "component": component,
            "tier": tier,
            "model": model.model,
            "prompt_tokens": str(prompt_tokens),
            "completion_tokens": str(completion_tokens),
            "latency_ms": f"{latency_ms:.3f}",
            "actual_cost_usd": f"{actual:.10f}",
            "flagship_equivalent_cost_usd": f"{equivalent:.10f}",
            "routing_savings_usd": f"{max(equivalent - actual, 0.0):.10f}",
        },
    )


def log_agent_diagnostic(
    *,
    outcome: str,
    synthesis_attempts: int,
    log_path: Path | None = None,
) -> None:
    """Persist a non-content answer-agent outcome for operational diagnosis."""
    if not outcome or synthesis_attempts < 0:
        raise ValueError("outcome must be non-empty and synthesis_attempts must be non-negative.")
    _persist(
        log_path,
        _DIAGNOSTIC_FILENAME,
        AGENT_DIAGNOSTIC_FIELDS,
        {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "outcome": outcome,
            "synthesis_attempts": str(synthesis_attempts),
        },
    )
