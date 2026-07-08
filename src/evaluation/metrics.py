"""Forecasting accuracy metrics: RMSE, MAE, MASE, R squared.

Pure numerical functions. Empty arrays, length mismatches, and NaN values
raise ValueError so the caller handles missingness explicitly. MASE follows
the Hyndman and Koehler 2006 definition with a non-seasonal naive
denominator (one-step-ahead).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def compute_rmse(y_true, y_pred) -> float:
    """Returns the root mean squared error of y_pred against y_true."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    _validate_pair(y_true_arr, y_pred_arr)
    return float(root_mean_squared_error(y_true_arr, y_pred_arr))


def compute_mae(y_true, y_pred) -> float:
    """Returns the mean absolute error of y_pred against y_true."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    _validate_pair(y_true_arr, y_pred_arr)
    return float(mean_absolute_error(y_true_arr, y_pred_arr))


def compute_mase(y_true, y_pred, y_train) -> float:
    """Returns the mean absolute scaled error of y_pred against y_true, scaled by the in-sample naive one-step-ahead error on y_train.

    Hyndman and Koehler 2006. MASE less than 1 means the forecast beats the
    naive baseline; MASE greater than 1 means it is worse.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_train_arr = np.asarray(y_train)
    _validate_pair(y_true_arr, y_pred_arr)
    if y_train_arr.size == 0 or np.isnan(y_train_arr).any():
        raise ValueError("y_train must be non-empty and contain no NaN")
    if y_train_arr.size < 2:
        raise ValueError("y_train needs at least 2 points to compute the naive denominator")
    forecast_mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
    naive_mae = float(np.mean(np.abs(np.diff(y_train_arr))))
    if naive_mae == 0:
        raise ValueError("y_train naive denominator is 0 (constant series); MASE undefined")
    return forecast_mae / naive_mae


def compute_r2(y_true, y_pred) -> float:
    """Returns the R squared (coefficient of determination) of y_pred against y_true."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    _validate_pair(y_true_arr, y_pred_arr)
    return float(r2_score(y_true_arr, y_pred_arr))


def compute_all_metrics(y_true, y_pred, y_train) -> dict[str, float]:
    """Returns all four metrics (rmse, mae, mase, r2) as a dict."""
    return {
        "rmse": compute_rmse(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
        "mase": compute_mase(y_true, y_pred, y_train),
        "r2": compute_r2(y_true, y_pred),
    }


def _validate_pair(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Raises ValueError if the arrays are empty, mismatched in length, or contain NaN."""
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("y_true and y_pred must be non-empty")
    if y_true.size != y_pred.size:
        raise ValueError(f"length mismatch: len(y_true)={y_true.size}, len(y_pred)={y_pred.size}")
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("y_true and y_pred must contain no NaN values")
