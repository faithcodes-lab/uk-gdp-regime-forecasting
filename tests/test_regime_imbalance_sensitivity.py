"""Tests for scripts/regime_imbalance_sensitivity.py.

Only leave_one_quarter_out_rmse() is unit tested here since it is pure
and fast; the full check against real predictions is covered by the
script's own main().
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.regime_imbalance_sensitivity import leave_one_quarter_out_rmse


def _synthetic_group(n: int = 6, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "quarter": pd.period_range("2020-01-01", periods=n, freq="Q").to_timestamp(),
            "y_true": rng.normal(size=n),
            "y_pred": rng.normal(size=n),
        }
    )


def test_leave_one_quarter_out_returns_one_row_per_quarter():
    """For n quarters, there are exactly n leave-one-out rows."""
    group = _synthetic_group(n=6)
    result = leave_one_quarter_out_rmse(group)
    assert len(result) == 6


def test_leave_one_quarter_out_full_rmse_is_constant_across_rows():
    """full_regime_rmse is the same value in every row, since it does not depend on which quarter is held out."""
    group = _synthetic_group(n=6)
    result = leave_one_quarter_out_rmse(group)
    assert result["full_regime_rmse"].nunique() == 1


def test_leave_one_quarter_out_swing_zero_for_identical_predictions():
    """A perfect-prediction group has RMSE 0 throughout, so every swing is exactly 0."""
    group = _synthetic_group(n=6)
    group["y_pred"] = group["y_true"]
    result = leave_one_quarter_out_rmse(group)
    assert (result["swing"] == 0.0).all()


def test_leave_one_quarter_out_dropping_the_worst_quarter_lowers_rmse():
    """Dropping the quarter with the single largest error reduces leave-one-out RMSE below the full value."""
    group = pd.DataFrame(
        {
            "quarter": pd.period_range("2020-01-01", periods=4, freq="Q").to_timestamp(),
            "y_true": [0.0, 0.0, 0.0, 0.0],
            "y_pred": [0.1, 0.1, 0.1, 10.0],
        }
    )
    result = leave_one_quarter_out_rmse(group)
    worst_quarter = group["quarter"].iloc[3]
    row = result.loc[result["held_out_quarter"] == worst_quarter].iloc[0]
    assert row["leave_one_out_rmse"] < row["full_regime_rmse"]
    assert row["swing"] < 0
