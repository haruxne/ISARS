"""Public implementation of interval selection for adversarial rank suppression."""

from .interval_selection import IntervalSelection, select_intervals
from .results import load_reported_results, verify_reported_results

__all__ = [
    "IntervalSelection",
    "load_reported_results",
    "select_intervals",
    "verify_reported_results",
]
