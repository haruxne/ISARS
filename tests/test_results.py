from pathlib import Path

from isars.results import format_summary, load_reported_results, verify_reported_results

RESULTS = Path(__file__).parents[1] / "reproduction" / "reported_results.json"
PACKAGED_RESULTS = Path(__file__).parents[1] / "src" / "isars" / "reported_results.json"


def test_reported_sign_tests_recompute_exactly() -> None:
    verified = verify_reported_results(load_reported_results(RESULTS))
    assert verified["batch_a_extension"]["target_centered"].wins == 61
    assert verified["batch_a_extension"]["highest_score_window"].losses == 0


def test_summary_contains_primary_metrics() -> None:
    summary = format_summary(load_reported_results(RESULTS))
    assert "minimum_unaffected_floor" in summary
    assert "65.91" in summary
    assert "32/58" in summary


def test_packaged_results_match_reproduction_artifact() -> None:
    assert PACKAGED_RESULTS.read_bytes() == RESULTS.read_bytes()
