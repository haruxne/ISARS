import pytest

from isars.interval_selection import IntervalSelectionError, build_sliding_windows, select_intervals


def test_sliding_windows_cover_the_final_valid_start() -> None:
    windows = build_sliding_windows(20, window_size=4, stride=3)
    assert windows[-1] == (16, 17, 18, 19)


def test_select_intervals_uses_earliest_start_as_tie_break() -> None:
    scores = [0.5] * 9
    selected = select_intervals(
        scores,
        frame_count=10,
        target_position=4,
        window_size=2,
        stride=1,
    )
    assert selected["target_centered"].start == 4
    assert selected["highest_score_window"].start == 0
    assert selected["minimum_unaffected_floor"].start == 0


def test_score_count_must_match_window_count() -> None:
    with pytest.raises(IntervalSelectionError, match="windows are required"):
        select_intervals([0.1], frame_count=8, target_position=2, window_size=4)


def test_scores_must_be_finite() -> None:
    with pytest.raises(IntervalSelectionError, match="finite numbers"):
        select_intervals(
            [0.1, 0.2, float("nan"), 0.4, 0.5],
            frame_count=8,
            target_position=2,
            window_size=4,
        )
