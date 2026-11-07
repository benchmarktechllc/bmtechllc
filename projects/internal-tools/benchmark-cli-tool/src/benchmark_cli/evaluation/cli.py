"""Command-line runner for retrieval-only golden-dataset evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..agent import run_agent
from ..config import EVALUATION_LOG_PATH, GOLDEN_DATASET_PATH
from ..retrieval import search_docs, vector_only_search
from .calibration import calibrate_abstention
from .dataset import load_golden_dataset, resolve_golden_dataset
from .metrics import (
    compare_retrieval_modes,
    comparison_markdown,
    evaluate_answers,
    evaluate_retrieval,
)
from .reporting import summarize_cost_rows, usage_rows
from .telemetry import append_evaluation_log


def _search_including_fixtures(query: str, top_k: int = 5, **kwargs: object):
    """Golden-set measurement still sees synthetic fixtures. Ordinary answers do not."""
    kwargs["include_synthetic"] = True
    return search_docs(query, top_k, **kwargs)  # type: ignore[arg-type]


def _vector_including_fixtures(query: str, top_k: int = 5, **kwargs: object):
    kwargs["include_synthetic"] = True
    return vector_only_search(query, top_k, **kwargs)  # type: ignore[arg-type]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=GOLDEN_DATASET_PATH)
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=EVALUATION_LOG_PATH,
        help=f"Append per-query evaluation metrics to this CSV (default: {EVALUATION_LOG_PATH}).",
    )
    parser.add_argument(
        "--calibrate-abstention",
        action="store_true",
        help="Sweep configured evidence cutoffs and report the selected pair.",
    )
    parser.add_argument(
        "--answers",
        action="store_true",
        help="Evaluate routed and forced-flagship grounded answers over the golden set.",
    )
    parser.add_argument(
        "--answer-report",
        type=Path,
        help="Write routed-versus-flagship answer metrics to this JSON file.",
    )
    parser.add_argument(
        "--compare-vector",
        action="store_true",
        help="Measure vector-only and hybrid RRF retrieval against the same golden dataset.",
    )
    parser.add_argument(
        "--comparison-report",
        type=Path,
        help="Write the vector-only versus hybrid Recall@5/MRR table to this Markdown file.",
    )
    args = parser.parse_args()
    if args.comparison_report and not args.compare_vector:
        parser.error("--comparison-report requires --compare-vector")
    if args.answer_report and not args.answers:
        parser.error("--answer-report requires --answers")
    cases = resolve_golden_dataset(load_golden_dataset(args.dataset))
    if args.compare_vector:
        comparison = compare_retrieval_modes(
            cases,
            vector_search=_vector_including_fixtures,
            hybrid_search=_search_including_fixtures,
        )
        append_evaluation_log(
            args.metrics_csv,
            args.dataset,
            comparison.vector_only,
            retrieval_mode="vector_only",
        )
        append_evaluation_log(
            args.metrics_csv,
            args.dataset,
            comparison.hybrid,
            retrieval_mode="hybrid_rrf",
        )
        payload = comparison.as_dict()
        if args.comparison_report:
            args.comparison_report.parent.mkdir(parents=True, exist_ok=True)
            args.comparison_report.write_text(comparison_markdown(comparison) + "\n", encoding="utf-8")
            payload["comparison_report"] = str(args.comparison_report)
    else:
        summary = evaluate_retrieval(cases, _search_including_fixtures)
        append_evaluation_log(args.metrics_csv, args.dataset, summary, retrieval_mode="hybrid_rrf")
        payload = summary.as_dict()
    if args.answers:
        before_rows = usage_rows()
        routed = evaluate_answers(cases, lambda question, **filters: run_agent(question, **filters))
        routed_rows = usage_rows()
        flagship = evaluate_answers(
            cases, lambda question, **filters: run_agent(question, tier="flagship", **filters)
        )
        flagship_rows = usage_rows()
        payload["answers"] = {
            "routed": routed.as_dict(),
            "flagship_only": flagship.as_dict(),
            "cost_breakdown": {
                "routed": summarize_cost_rows(routed_rows[len(before_rows) :]),
                "flagship_only": summarize_cost_rows(flagship_rows[len(routed_rows) :]),
            },
        }
        if args.answer_report:
            args.answer_report.parent.mkdir(parents=True, exist_ok=True)
            args.answer_report.write_text(json.dumps(payload["answers"], indent=2) + "\n", encoding="utf-8")
            payload["answer_report"] = str(args.answer_report)
    if args.calibrate_abstention:
        payload["abstention_calibration"] = calibrate_abstention(cases, _search_including_fixtures).as_dict()
    payload["metrics_csv"] = str(args.metrics_csv)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
