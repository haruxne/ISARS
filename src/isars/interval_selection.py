"""Contiguous interval-selection strategies used by ISARS."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


class IntervalSelectionError(ValueError):
    """Raised when interval-selection inputs violate the protocol."""


@dataclass(frozen=True)
class IntervalSelection:
    """One selected contiguous interval and its clean unaffected-score floor."""

    start: int
    positions: tuple[int, ...]
    unaffected_score_floor: float | None
    unaffected_window_count: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["positions"] = list(self.positions)
        return payload


def build_sliding_windows(
    frame_count: int,
    *,
    window_size: int = 16,
    stride: int = 1,
) -> tuple[tuple[int, ...], ...]:
    """Create deterministic full-coverage windows over original frame positions."""

    for name, value in {
        "frame_count": frame_count,
        "window_size": window_size,
        "stride": stride,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise IntervalSelectionError(f"{name} must be a positive integer")
    if frame_count < window_size:
        raise IntervalSelectionError("frame_count must be at least window_size")
    if stride > window_size:
        raise IntervalSelectionError("stride must not exceed window_size")

    final_start = frame_count - window_size
    starts = list(range(0, final_start + 1, stride))
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(tuple(range(start, start + window_size)) for start in starts)


def target_centered_start(
    frame_count: int,
    target_position: int,
    *,
    window_size: int = 16,
) -> int:
    """Return the clamped start of the interval centered on the target frame."""

    if isinstance(target_position, bool) or not isinstance(target_position, int):
        raise IntervalSelectionError("target_position must be an integer")
    if not 0 <= target_position < frame_count:
        raise IntervalSelectionError("target_position is outside the video")
    frames_before_target = (window_size - 1) // 2
    return min(max(target_position - frames_before_target, 0), frame_count - window_size)


def _evaluate_interval(
    start: int,
    *,
    scores: Sequence[float],
    windows: Sequence[Sequence[int]],
    window_size: int,
) -> IntervalSelection:
    unaffected = [
        index
        for index, window in enumerate(windows)
        if window[-1] < start or window[0] >= start + window_size
    ]
    floor = max((float(scores[index]) for index in unaffected), default=None)
    return IntervalSelection(
        start=start,
        positions=tuple(range(start, start + window_size)),
        unaffected_score_floor=floor,
        unaffected_window_count=len(unaffected),
    )


def select_intervals(
    scores: Sequence[float],
    *,
    frame_count: int,
    target_position: int,
    window_size: int = 16,
    stride: int = 1,
) -> dict[str, IntervalSelection]:
    """Select the three matched-budget intervals evaluated in the paper.

    ``minimum_unaffected_floor`` exhaustively evaluates every contiguous interval
    with the fixed frame budget. It minimizes the maximum clean score among windows
    that do not overlap the editable interval. Ties are resolved by the earliest
    interval start.
    """

    windows = build_sliding_windows(
        frame_count,
        window_size=window_size,
        stride=stride,
    )
    if len(scores) != len(windows):
        raise IntervalSelectionError(
            f"scores contains {len(scores)} values, but {len(windows)} windows are required"
        )
    if not scores:
        raise IntervalSelectionError("scores must not be empty")
    if any(
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        for score in scores
    ):
        raise IntervalSelectionError("scores must contain only finite numbers")

    centered_start = target_centered_start(
        frame_count,
        target_position,
        window_size=window_size,
    )
    highest_index = max(range(len(scores)), key=lambda index: float(scores[index]))
    highest_start = min(windows[highest_index][0], frame_count - window_size)
    candidates = [
        _evaluate_interval(
            start,
            scores=scores,
            windows=windows,
            window_size=window_size,
        )
        for start in range(frame_count - window_size + 1)
    ]
    minimum_floor = min(
        candidates,
        key=lambda item: (
            float("-inf") if item.unaffected_score_floor is None else item.unaffected_score_floor,
            item.start,
        ),
    )
    return {
        "target_centered": _evaluate_interval(
            centered_start,
            scores=scores,
            windows=windows,
            window_size=window_size,
        ),
        "highest_score_window": _evaluate_interval(
            highest_start,
            scores=scores,
            windows=windows,
            window_size=window_size,
        ),
        "minimum_unaffected_floor": minimum_floor,
    }
