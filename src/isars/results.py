"""Loading, validation, and display helpers for reported aggregate results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .statistics import SignTestResult, verify_sign_test


def load_reported_results(path: str | Path) -> dict[str, Any]:
    """Load the aggregate report and enforce its top-level evidence guards."""

    source = Path(path)
    try:
        report = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read reported results from {source}: {error}") from error
    if not isinstance(report, dict):
        raise ValueError("reported results must be a JSON object")
    if report.get("successful_execution_count") != 423 or report.get("failure_count") != 0:
        raise ValueError("unexpected execution counts in reported results")
    return report


def verify_reported_results(report: dict[str, Any]) -> dict[str, dict[str, SignTestResult]]:
    """Verify recorded paired sign tests for the primary and pooled cohorts."""

    verified: dict[str, dict[str, SignTestResult]] = {}
    for cohort_name in ("batch_a_extension", "pooled_141_secondary"):
        cohort = report[cohort_name]
        cohort_results: dict[str, SignTestResult] = {}
        for baseline, comparison in cohort["paired_comparisons"].items():
            wins, ties, losses = comparison["wins_ties_losses"]
            if wins + ties + losses != cohort["query_count"]:
                raise ValueError(f"{cohort_name}/{baseline}: paired count mismatch")
            cohort_results[baseline] = verify_sign_test(
                wins,
                ties,
                losses,
                expected_p_value=comparison["exact_two_sided_sign_test_p_unadjusted"],
                expected_adjusted_p_value=comparison["bonferroni_adjusted_p"],
                family_size=comparison["bonferroni_family_size"],
            )
        verified[cohort_name] = cohort_results
    return verified


def format_summary(report: dict[str, Any]) -> str:
    """Format the primary 93-query results as a compact text table."""

    cohort = report["batch_a_extension"]
    rows = [
        "strategy                 mean_rank   MRR    R@10   ASR@10",
        "-----------------------  ---------  -----  -----  ------",
    ]
    for strategy in (
        "target_centered",
        "highest_score_window",
        "minimum_unaffected_floor",
    ):
        values = cohort["metrics"][strategy]
        asr = f"{values['asr_at_10_numerator']}/{values['asr_at_10_denominator']}"
        rows.append(
            f"{strategy:23}  {values['mean_rank']:9.2f}  {values['mrr']:.3f}  "
            f"{values['recall_at_10']:.3f}  {asr:>6}"
        )
    return "\n".join(rows)
