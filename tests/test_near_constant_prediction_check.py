"""Tests for scripts/near_constant_prediction_check.py.

Only constant_mean_comparison() is unit tested here since it is pure
and fast; same_terminal_leaves() and the full check need a real fitted
XGBoost model and are covered by the script's own main().
"""

from __future__ import annotations

import numpy as np

from scripts.near_constant_prediction_check import constant_mean_comparison


def test_constant_mean_comparison_counts_distinct_predictions():
    """A prediction array with two repeated values is reported as 2 distinct predictions."""
    y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y_pred = np.array([0.44, 0.44, 0.44, 0.44, 0.10])
    stats = constant_mean_comparison(y_true, y_pred, train_mean=0.3)
    assert stats["n_distinct_predictions"] == 2
    assert stats["most_common_count"] == 4


def test_constant_mean_comparison_zero_improvement_for_constant_predictor():
    """A predictor that always outputs the training mean has 0% improvement over that same benchmark."""
    y_true = np.array([0.1, 0.5, -0.2, 0.3])
    train_mean = 0.2
    y_pred = np.full_like(y_true, train_mean)
    stats = constant_mean_comparison(y_true, y_pred, train_mean)
    assert stats["improvement_pct"] == 0.0
    assert stats["model_rmse"] == stats["constant_mean_rmse"]


def test_constant_mean_comparison_full_improvement_for_perfect_predictor():
    """A predictor that exactly matches y_true has 100% improvement and zero model RMSE."""
    y_true = np.array([0.1, 0.5, -0.2, 0.3])
    stats = constant_mean_comparison(y_true, y_true.copy(), train_mean=0.175)
    assert stats["model_rmse"] == 0.0
    assert stats["improvement_pct"] == 100.0
