"""Tests for src/evaluation/metrics.py.

The MASE hand-computed example and the NaN and empty-array guards are the
parts that matter most: MASE has no sklearn parity to lean on, and silent
NaN handling would propagate undetected through downstream evaluation.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from src.evaluation.metrics import (
    compute_all_metrics,
    compute_mae,
    compute_mase,
    compute_r2,
    compute_rmse,
)


def test_rmse_zero_when_prediction_equals_truth():
    """RMSE is 0 when predictions exactly match the truth."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert compute_rmse(y, y) == 0.0


def test_rmse_matches_sklearn():
    """compute_rmse equals sklearn root_mean_squared_error."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    assert compute_rmse(y_true, y_pred) == pytest.approx(root_mean_squared_error(y_true, y_pred))


def test_mae_zero_when_prediction_equals_truth():
    """MAE is 0 when predictions exactly match the truth."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert compute_mae(y, y) == 0.0


def test_mae_matches_sklearn():
    """compute_mae equals sklearn mean_absolute_error."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    assert compute_mae(y_true, y_pred) == pytest.approx(mean_absolute_error(y_true, y_pred))


def test_r2_one_for_perfect_prediction():
    """R squared is 1.0 when predictions exactly match the truth."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert compute_r2(y, y) == 1.0


def test_r2_matches_sklearn():
    """compute_r2 equals sklearn r2_score."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    assert compute_r2(y_true, y_pred) == pytest.approx(r2_score(y_true, y_pred))


def test_mase_zero_when_prediction_equals_truth():
    """MASE is 0 when forecast errors are zero."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    y_train = np.array([0.5, 1.5, 2.5])
    assert compute_mase(y_true, y_pred, y_train) == 0.0


def test_mase_hand_computed_example():
    """MASE on a small array equals the hand-computed value.

    y_train = [0, 1, 2, 3, 4]; naive MAE = mean(|1-0|, |2-1|, |3-2|, |4-3|) = 1.0.
    y_true = [5, 6, 7], y_pred = [5.5, 6.5, 7.5]; forecast MAE = mean(0.5, 0.5, 0.5) = 0.5.
    MASE = 0.5 / 1.0 = 0.5.
    """
    y_train = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_true = np.array([5.0, 6.0, 7.0])
    y_pred = np.array([5.5, 6.5, 7.5])
    assert compute_mase(y_true, y_pred, y_train) == pytest.approx(0.5)


def test_mase_less_than_one_for_better_than_naive():
    """A forecast that does better than the naive one-step-ahead returns MASE less than 1."""
    # noisy training so the naive denominator is large
    y_train = np.array([0.0, 5.0, 0.0, 5.0])
    # forecasts very close to truth so the numerator is small
    y_true = np.array([2.5, 2.5])
    y_pred = np.array([2.6, 2.4])
    assert compute_mase(y_true, y_pred, y_train) < 1.0


def test_mase_greater_than_one_for_worse_than_naive():
    """A forecast that does worse than the naive one-step-ahead returns MASE greater than 1."""
    # smooth training so the naive denominator is tiny
    y_train = np.array([0.0, 0.1, 0.2, 0.3])
    # forecasts far from truth so the numerator is huge
    y_true = np.array([0.4, 0.5, 0.6])
    y_pred = np.array([100.0, 100.0, 100.0])
    assert compute_mase(y_true, y_pred, y_train) > 1.0


def test_compute_all_metrics_returns_all_four_keys():
    """compute_all_metrics returns a dict with rmse, mae, mase, r2 as floats."""
    y_train = np.array([0.0, 1.0, 2.0])
    y_true = np.array([3.0, 4.0])
    y_pred = np.array([3.1, 3.9])
    result = compute_all_metrics(y_true, y_pred, y_train)
    assert set(result.keys()) == {"rmse", "mae", "mase", "r2"}
    for v in result.values():
        assert isinstance(v, float)


def test_empty_arrays_raise():
    """Empty y_true or y_pred raises ValueError, not silent NaN."""
    with pytest.raises(ValueError, match="non-empty"):
        compute_rmse(np.array([]), np.array([]))


def test_nan_in_y_true_raises():
    """NaN in y_true raises ValueError, not silent propagation."""
    with pytest.raises(ValueError, match="NaN"):
        compute_rmse(np.array([1.0, np.nan]), np.array([1.0, 2.0]))


def test_nan_in_y_pred_raises():
    """NaN in y_pred raises ValueError, not silent propagation."""
    with pytest.raises(ValueError, match="NaN"):
        compute_mae(np.array([1.0, 2.0]), np.array([1.0, np.nan]))


def test_length_mismatch_raises():
    """Length mismatch between y_true and y_pred raises ValueError."""
    with pytest.raises(ValueError, match="length mismatch"):
        compute_r2(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_mase_empty_y_train_raises():
    """Empty y_train raises ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        compute_mase(np.array([1.0]), np.array([1.0]), np.array([]))


def test_mase_short_y_train_raises():
    """y_train with fewer than 2 points raises ValueError (no naive denominator)."""
    with pytest.raises(ValueError, match="at least 2"):
        compute_mase(np.array([1.0]), np.array([1.0]), np.array([0.0]))


def test_mase_constant_y_train_raises():
    """y_train with zero naive MAE (constant series) raises ValueError (MASE undefined)."""
    with pytest.raises(ValueError, match="MASE undefined"):
        compute_mase(np.array([5.0]), np.array([4.5]), np.array([2.0, 2.0, 2.0]))


def test_mase_nan_in_y_train_raises():
    """NaN in y_train raises ValueError."""
    with pytest.raises(ValueError, match="non-empty and contain no NaN"):
        compute_mase(np.array([1.0]), np.array([1.0]), np.array([0.0, np.nan, 1.0]))
