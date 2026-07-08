"""Tests for src/regimes/volatility.py (ICSS variance break test)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.regimes.volatility import ICSSResult, icss_test


def _dates(n: int, start: str = "2000-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq="QS")


def _stationary_series(n: int = 100, seed: int = 42, sigma: float = 1.0) -> pd.Series:
    """Constant-variance Gaussian noise."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(scale=sigma, size=n), index=_dates(n), name="signal")


def _series_with_variance_shifts(
    sigmas: list[float],
    breakpoints: list[int],
    seed: int = 42,
    tail: int = 50,
) -> pd.Series:
    """Build a series with the given segment standard deviations.

    ``sigmas[k]`` is the standard deviation of segment ``k``. There must
    be one more sigma than there are breakpoints. The mean is zero
    throughout so only the variance shifts. The final segment runs for
    ``tail`` points after the last breakpoint.
    """
    assert len(sigmas) == len(breakpoints) + 1
    rng = np.random.default_rng(seed)
    segment_starts = [0] + list(breakpoints)
    values: list[float] = []
    for k, start in enumerate(segment_starts):
        end = breakpoints[k] if k < len(breakpoints) else start + tail
        size = end - start
        values.extend(rng.normal(scale=sigmas[k], size=size).tolist())
    arr = np.array(values, dtype=float)
    return pd.Series(arr, index=_dates(len(arr)), name="signal")


def test_constant_variance_returns_no_breaks():
    """A stationary Gaussian signal should yield zero variance breaks at 5%."""
    series = _stationary_series(n=120, seed=7)
    result = icss_test(series, alpha=0.05)
    assert result.n_breaks == 0
    assert result.breakpoint_dates == []
    assert result.d_statistics == []


def test_known_single_variance_shift_detected():
    """A clear sigma=1 to sigma=4 shift at index 60 should be detected near 60."""
    series = _series_with_variance_shifts(sigmas=[1.0, 4.0], breakpoints=[60], seed=42, tail=60)
    result = icss_test(series, alpha=0.05)
    assert result.n_breaks >= 1
    detected_positions = [series.index.get_loc(d) for d in result.breakpoint_dates]
    assert min(abs(p - 60) for p in detected_positions) <= 5


def test_multiple_known_variance_shifts_detected():
    """sigma sequence [1, 4, 1] across two breaks should be detected."""
    series = _series_with_variance_shifts(
        sigmas=[1.0, 4.0, 1.0], breakpoints=[60, 130], seed=7, tail=70
    )
    result = icss_test(series, alpha=0.05)
    assert result.n_breaks >= 2
    detected_positions = [series.index.get_loc(d) for d in result.breakpoint_dates]
    for true_bp in [60, 130]:
        assert min(abs(p - true_bp) for p in detected_positions) <= 5


def test_breakpoint_date_is_a_real_index_value():
    """Each returned date must be an exact element of ``series.index``."""
    series = _series_with_variance_shifts(sigmas=[1.0, 5.0], breakpoints=[80], seed=11, tail=80)
    result = icss_test(series, alpha=0.05)
    for date in result.breakpoint_dates:
        assert date in series.index


def test_short_series_returns_no_breaks():
    """A series shorter than the minimum returns [] and emits a warning."""
    series = _stationary_series(n=10, seed=0)
    result = icss_test(series, alpha=0.05)
    assert result.n_breaks == 0
    assert result.breakpoint_dates == []


def test_non_datetime_index_raises():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])  # RangeIndex
    with pytest.raises(ValueError, match="DatetimeIndex"):
        icss_test(series, alpha=0.05)


def test_nan_values_dropped_before_test():
    series = _series_with_variance_shifts(sigmas=[1.0, 4.0], breakpoints=[60], seed=42, tail=60)
    contaminated = series.copy()
    contaminated.iloc[5] = np.nan
    contaminated.iloc[10] = np.nan
    result = icss_test(contaminated, alpha=0.05)
    assert result.n_observations == len(series) - 2


def test_alpha_changes_critical_value():
    """Different alphas should map to the published Inclan-Tiao critical values."""
    series = _stationary_series(n=60)
    r01 = icss_test(series, alpha=0.01)
    r05 = icss_test(series, alpha=0.05)
    r10 = icss_test(series, alpha=0.10)
    assert r01.critical_value == 1.628
    assert r05.critical_value == 1.358
    assert r10.critical_value == 1.224
    # The same series at stricter alpha should not return more breaks.
    assert r01.n_breaks <= r05.n_breaks <= r10.n_breaks


def test_invalid_alpha_raises():
    series = _stationary_series(n=60)
    with pytest.raises(ValueError, match="alpha"):
        icss_test(series, alpha=0.5)


def test_returns_icssresult_with_correct_types():
    series = _series_with_variance_shifts(sigmas=[1.0, 4.0], breakpoints=[60], seed=42, tail=60)
    result = icss_test(series, alpha=0.05)
    assert isinstance(result, ICSSResult)
    assert isinstance(result.method, str)
    assert isinstance(result.series_name, str)
    assert isinstance(result.n_observations, int)
    assert isinstance(result.significance_level, float)
    assert isinstance(result.critical_value, float)
    assert isinstance(result.breakpoint_dates, list)
    assert isinstance(result.d_statistics, list)
    assert isinstance(result.n_breaks, int)
    if result.breakpoint_dates:
        assert isinstance(result.breakpoint_dates[0], pd.Timestamp)
        assert isinstance(result.d_statistics[0], float)
    assert result.n_breaks == len(result.breakpoint_dates)
    assert result.n_breaks == len(result.d_statistics)
    assert result.method.startswith("ICSS")
