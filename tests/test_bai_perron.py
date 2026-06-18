"""Tests for src/regimes/bai_perron.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.regimes.bai_perron import (
    detect_breaks_binseg,
    detect_breaks_pelt,
    tune_penalty,
)


def _stationary_signal(n: int = 100, seed: int = 42) -> pd.Series:
    """Gaussian noise around a constant mean, with a quarterly DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n, freq="QS")
    return pd.Series(rng.normal(scale=1.0, size=n), index=dates, name="signal")


def _signal_with_breaks(
    true_break_indices: list[int],
    segment_means: list[float],
    n: int = 100,
    noise: float = 0.5,
    seed: int = 42,
) -> pd.Series:
    """Signal made of segments with given means and a known break pattern.

    ``segment_means[k]`` is the mean of segment ``k``; there must be one
    more mean than there are breakpoints.
    """
    assert len(segment_means) == len(true_break_indices) + 1
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n, freq="QS")
    values = np.zeros(n)
    segment_starts = [0] + list(true_break_indices)
    segment_ends = list(true_break_indices) + [n]
    for k, (start, end) in enumerate(zip(segment_starts, segment_ends)):
        values[start:end] = segment_means[k] + \
            rng.normal(scale=noise, size=end - start)
    return pd.Series(values, index=dates, name="signal")


def _nearest_distance(detected_idx: int, true_indices: list[int]) -> int:
    """Smallest absolute distance from a detected index to any true break."""
    return min(abs(detected_idx - t) for t in true_indices)


def test_pelt_detects_no_break_in_stationary_signal():
    """A stationary Gaussian signal at a high penalty should yield zero breaks."""
    series = _stationary_signal(n=100)
    breaks = detect_breaks_pelt(series, penalty=100.0)
    assert breaks == []


def test_pelt_detects_known_single_break():
    """A single mean shift should be detected near the true index."""
    series = _signal_with_breaks(
        true_break_indices=[50], segment_means=[0.0, 5.0], n=100)
    breaks = detect_breaks_pelt(series, penalty=10.0)
    assert len(breaks) >= 1
    detected_indices = [series.index.get_loc(b) for b in breaks]
    assert min(_nearest_distance(idx, [50]) for idx in detected_indices) <= 3


def test_pelt_detects_known_two_breaks():
    """Two mean shifts at known indices should both be detected within tolerance."""
    series = _signal_with_breaks(
        true_break_indices=[33, 66],
        segment_means=[0.0, 5.0, -3.0],
        n=100,
    )
    breaks = detect_breaks_pelt(series, penalty=10.0)
    assert len(breaks) >= 2
    detected_indices = [series.index.get_loc(b) for b in breaks]
    # Each true break should have a detected break within tolerance.
    for true_idx in [33, 66]:
        assert min(abs(idx - true_idx) for idx in detected_indices) <= 3


def test_pelt_detects_many_breaks_at_low_penalty():
    """Many regimes + low penalty should yield at least as many breaks."""
    true_breaks = [16, 33, 50, 66, 83]
    series = _signal_with_breaks(
        true_break_indices=true_breaks,
        segment_means=[0.0, 5.0, 0.0, -5.0, 0.0, 5.0],
        n=100,
    )
    breaks = detect_breaks_pelt(series, penalty=2.0)
    assert len(breaks) >= 5


def test_binseg_returns_exact_number_of_breaks():
    """Binary Segmentation returns exactly the requested number of breaks."""
    series = _signal_with_breaks(
        true_break_indices=[33, 66],
        segment_means=[0.0, 5.0, -3.0],
        n=100,
    )
    breaks = detect_breaks_binseg(series, n_breaks=2)
    assert len(breaks) == 2


def test_binseg_zero_breaks_returns_empty():
    """Asking for zero breaks should return an empty list."""
    series = _stationary_signal(n=50)
    assert detect_breaks_binseg(series, n_breaks=0) == []


def test_binseg_rejects_negative_n_breaks():
    """A negative n_breaks should raise ValueError."""
    series = _stationary_signal(n=50)
    with pytest.raises(ValueError):
        detect_breaks_binseg(series, n_breaks=-1)


def test_short_series_returns_empty_list_for_pelt():
    """A series shorter than the minimum should return [] without crashing."""
    dates = pd.date_range("2000-01-01", periods=3, freq="QS")
    series = pd.Series([1.0, 2.0, 3.0], index=dates)
    assert detect_breaks_pelt(series, penalty=10.0) == []


def test_short_series_returns_empty_list_for_binseg():
    """BinSeg should also be silent on short input."""
    dates = pd.date_range("2000-01-01", periods=3, freq="QS")
    series = pd.Series([1.0, 2.0, 3.0], index=dates)
    assert detect_breaks_binseg(series, n_breaks=1) == []


def test_non_datetime_index_raises():
    """Both detectors require a DatetimeIndex."""
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])  # RangeIndex
    with pytest.raises(ValueError):
        detect_breaks_pelt(series, penalty=10.0)
    with pytest.raises(ValueError):
        detect_breaks_binseg(series, n_breaks=1)


def test_nan_values_dropped_before_detection():
    """NaN values are dropped; the remaining points are passed to the algorithm."""
    series = _signal_with_breaks(
        true_break_indices=[50], segment_means=[0.0, 5.0], n=100)
    series_with_nan = series.copy()
    series_with_nan.iloc[10] = np.nan
    series_with_nan.iloc[20] = np.nan
    breaks = detect_breaks_pelt(series_with_nan, penalty=10.0)
    # The break should still be detected near the true change point.
    if breaks:
        clean_index = series_with_nan.dropna().index
        detected_positions = [clean_index.get_loc(b) for b in breaks]
        # True break is at original index 50, which corresponds to clean index 48
        # (after dropping two earlier NaN rows).
        assert min(abs(p - 48) for p in detected_positions) <= 4


def test_break_index_maps_to_first_quarter_of_new_segment():
    """A break at index ``i`` should return ``series.index[i]`` exactly."""
    # Two clear segments with a strong shift right at index 50.
    series = _signal_with_breaks(
        true_break_indices=[50],
        segment_means=[0.0, 10.0],
        n=100,
        noise=0.1,
    )
    breaks = detect_breaks_pelt(series, penalty=20.0)
    assert len(breaks) == 1
    detected_idx = series.index.get_loc(breaks[0])
    # The detected index must be the position of one of the actual rows
    # of the series (not interpolated), and very close to 50 with low noise.
    assert breaks[0] == series.index[detected_idx]
    assert abs(detected_idx - 50) <= 2


def test_tune_penalty_returns_expected_columns():
    """tune_penalty's DataFrame has the documented schema."""
    series = _stationary_signal(n=80)
    df = tune_penalty(series, penalties=[10.0, 20.0])
    assert list(df.columns) == ["penalty", "n_breaks", "breakpoints"]
    assert len(df) == 2
    assert df["penalty"].dtype == float
    assert df["n_breaks"].dtype == int
    # Pandas may infer either ``object`` or ``string`` for the
    # breakpoints column; what matters is that the values are strings.
    assert all(isinstance(v, str) for v in df["breakpoints"])


def test_tune_penalty_uses_default_penalty_grid():
    """Default penalty grid is [5, 10, 15, 20, 30]."""
    series = _stationary_signal(n=80)
    df = tune_penalty(series)
    assert list(df["penalty"]) == [5.0, 10.0, 15.0, 20.0, 30.0]


def test_tune_penalty_breakpoints_column_format():
    """Breakpoints column is empty string for zero breaks, ISO-comma otherwise."""
    series = _signal_with_breaks(
        true_break_indices=[50], segment_means=[0.0, 10.0], n=100, noise=0.1
    )
    df = tune_penalty(series, penalties=[5.0, 1000.0])
    # Low penalty: at least one break detected; the string is a non-empty
    # comma-separated list of ISO dates.
    low_row = df[df["penalty"] == 5.0].iloc[0]
    assert low_row["n_breaks"] >= 1
    for piece in low_row["breakpoints"].split(", "):
        # Each piece parses as a date.
        pd.Timestamp(piece)
    # High penalty: zero breaks; empty string.
    high_row = df[df["penalty"] == 1000.0].iloc[0]
    assert high_row["n_breaks"] == 0
    assert high_row["breakpoints"] == ""
