"""Exact statistical checks used by the lightweight reproduction path."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SignTestResult:
    wins: int
    ties: int
    losses: int
    p_value: float
    adjusted_p_value: float


def exact_two_sided_sign_test(wins: int, losses: int) -> float:
    """Return the exact two-sided sign-test p-value, excluding ties."""

    if min(wins, losses) < 0:
        raise ValueError("wins and losses must be non-negative")
    sample_size = wins + losses
    if sample_size == 0:
        return 1.0
    tail = sum(math.comb(sample_size, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / 2**sample_size)


def verify_sign_test(
    wins: int,
    ties: int,
    losses: int,
    *,
    expected_p_value: float,
    expected_adjusted_p_value: float,
    family_size: int = 2,
) -> SignTestResult:
    """Recompute and verify one recorded sign test and Bonferroni correction."""

    if ties < 0 or family_size < 1:
        raise ValueError("ties must be non-negative and family_size must be positive")
    p_value = exact_two_sided_sign_test(wins, losses)
    adjusted = min(1.0, p_value * family_size)
    if not math.isclose(p_value, expected_p_value, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("recorded sign-test p-value does not match the exact calculation")
    if not math.isclose(adjusted, expected_adjusted_p_value, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("recorded adjusted p-value does not match Bonferroni correction")
    return SignTestResult(wins, ties, losses, p_value, adjusted)
