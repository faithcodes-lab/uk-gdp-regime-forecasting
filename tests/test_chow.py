"""Tests for src/regimes/chow.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.regimes.chow import (
    ChowTestResult,
    InsufficientObservationsError,
    chow_test,
)


def _synthetic_data(
    *,
    n: int = 100,
    shift: float = 0.0,
    breakpoint_idx: int = 50,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build synthetic X/y/dates with an optional mean shift at ``breakpoint_idx``.

    Dates are quarter-start so the synthetic series mimics the regime
    boundary basis. The chow_test function normalises to quarter-start
    internally, so feeding either quarter-start or quarter-end inputs
    produces the same split.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n, freq="QS")
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    y_values = 1.0 + 0.5 * X["x1"] + 0.3 * X["x2"] + rng.normal(scale=0.1, size=n)
    y_values = y_values.copy()
    y_values.iloc[breakpoint_idx:] += shift
    return X, pd.Series(y_values, name="y"), pd.Series(dates, name="date")


def test_significant_break_is_detected():
    """A large mean shift in the synthetic data should produce p < 0.05."""
    X, y, dates = _synthetic_data(n=100, shift=2.0, breakpoint_idx=50)
    result = chow_test(X, y, dates, breakpoint_date="2012-07-01")
    assert result.p_value < 0.05
    assert result.f_statistic > 0


def test_no_break_when_homogeneous():
    """No mean shift in the synthetic data should produce p > 0.05."""
    X, y, dates = _synthetic_data(n=100, shift=0.0, breakpoint_idx=50)
    result = chow_test(X, y, dates, breakpoint_date="2012-07-01")
    assert result.p_value > 0.05


def test_result_has_all_fields_with_correct_types():
    """The dataclass should expose all 12 fields with the documented types."""
    X, y, dates = _synthetic_data(n=100)
    result = chow_test(X, y, dates, breakpoint_date="2012-07-01")
    assert isinstance(result, ChowTestResult)
    assert isinstance(result.breakpoint_date, pd.Timestamp)
    assert isinstance(result.f_statistic, float)
    assert isinstance(result.p_value, float)
    assert isinstance(result.rss_pooled, float)
    assert isinstance(result.rss_pre, float)
    assert isinstance(result.rss_post, float)
    assert isinstance(result.n_total, int)
    assert isinstance(result.n_pre, int)
    assert isinstance(result.n_post, int)
    assert isinstance(result.k_params, int)
    assert isinstance(result.df_numerator, int)
    assert isinstance(result.df_denominator, int)


def test_boundary_quarter_routes_to_post_when_dataset_is_quarter_end():
    """Key boundary-convention test.

    The project dataset uses quarter-end dates while regime boundaries
    are quarter-start. The chow_test must route the boundary quarter
    (e.g. Q1 2008 for a 2008-01-01 breakpoint) into the post-period.
    """
    n = 40
    quarter_periods = pd.period_range("2005Q1", periods=n, freq="Q")
    dates_qe = pd.Series(quarter_periods.to_timestamp(how="end").normalize())
    rng = np.random.default_rng(123)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    y = pd.Series(1.0 + rng.normal(scale=0.1, size=n))

    # Breakpoint is the quarter-start of the 20th quarter (2010-01-01).
    breakpoint = quarter_periods[20].to_timestamp(how="start")
    result = chow_test(X, y, dates_qe, breakpoint_date=breakpoint)

    # The 20th quarter (index 20, e.g. 2010 Q1, quarter-end 2010-03-31)
    # must land in the post-period because its quarter-start equals the
    # breakpoint.
    assert result.n_pre == 20
    assert result.n_post == 20


def test_nan_rows_are_dropped_before_splitting():
    """Rows with NaN in X, y, or dates should be removed prior to the split."""
    X, y, dates = _synthetic_data(n=50)
    X = X.copy()
    y = y.copy()
    X.iloc[0, 0] = np.nan
    y.iloc[1] = np.nan
    result = chow_test(X, y, dates, breakpoint_date="2007-01-01")
    assert result.n_total == 48
    assert result.n_pre + result.n_post == 48


def test_insufficient_observations_raises():
    """If either sub-sample lacks positive residual DOF, raise."""
    dates = pd.Series(pd.date_range("2000-01-01", periods=4, freq="QS"))
    X = pd.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0], "x2": [0.5, 0.6, 0.7, 0.8]})
    y = pd.Series([1.0, 1.1, 1.2, 1.3])
    with pytest.raises(InsufficientObservationsError):
        chow_test(X, y, dates, breakpoint_date="2000-07-01")


def test_breakpoint_date_accepts_string_and_timestamp():
    """The breakpoint can be passed as a string or a pd.Timestamp."""
    X, y, dates = _synthetic_data(n=100)
    result_str = chow_test(X, y, dates, breakpoint_date="2012-07-01")
    result_ts = chow_test(X, y, dates, breakpoint_date=pd.Timestamp("2012-07-01"))
    assert result_str.f_statistic == pytest.approx(result_ts.f_statistic)
    assert result_str.p_value == pytest.approx(result_ts.p_value)


def test_k_params_includes_intercept():
    """k_params must equal the number of features plus one for the intercept."""
    X, y, dates = _synthetic_data(n=100)
    assert X.shape[1] == 2
    result = chow_test(X, y, dates, breakpoint_date="2012-07-01")
    assert result.k_params == 3


def test_degrees_of_freedom_match_formula():
    """df_numerator = k_params; df_denominator = n_total - 2 * k_params."""
    X, y, dates = _synthetic_data(n=100)
    result = chow_test(X, y, dates, breakpoint_date="2012-07-01")
    assert result.df_numerator == result.k_params
    assert result.df_denominator == result.n_total - 2 * result.k_params


def test_breakpoint_passed_as_quarter_end_still_routes_quarter_to_post():
    """Even when the breakpoint is given as a quarter-end date, the same
    quarter still routes to post.

    chow_test normalises the breakpoint to its quarter-start, so passing
    2008-03-31 (Q1 2008 quarter-end) yields the same split as passing
    2008-01-01.
    """
    X, y, dates = _synthetic_data(n=100)
    result_qs = chow_test(X, y, dates, breakpoint_date="2008-01-01")
    result_qe = chow_test(X, y, dates, breakpoint_date="2008-03-31")
    assert result_qs.n_pre == result_qe.n_pre
    assert result_qs.n_post == result_qe.n_post
    assert result_qs.f_statistic == pytest.approx(result_qe.f_statistic)
