"""Tests for src/evaluation/aggregation.py.

Tests that matter most: test_compute_per_fold_metrics_values_match_compute_all_metrics
(grouping correctness), and the mean/median/std-versus-numpy tests
(arithmetic correctness).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.aggregation import aggregate_cv_results, compute_per_fold_metrics
from src.evaluation.metrics import compute_all_metrics


def _synthetic_predictions_df(
    n_models: int = 4,
    n_schemes: int = 2,
    fold_sizes: list[int] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Builds synthetic predictions with custom fold sizes per scheme."""
    if fold_sizes is None:
        fold_sizes = [4, 4, 4]
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    models = ["ridge", "xgboost", "lightgbm", "arima"][:n_models]
    schemes = ["expanding_window", "regime_aligned"][:n_schemes]
    for model in models:
        for scheme in schemes:
            for fold_idx, size in enumerate(fold_sizes, start=1):
                for i in range(size):
                    rows.append(
                        {
                            "model": model,
                            "quarter": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i * 90),
                            "regime": "A",
                            "y_true": float(rng.normal()),
                            "y_pred": float(rng.normal()),
                            "fold_idx": fold_idx,
                            "scheme": scheme,
                        }
                    )
    return pd.DataFrame(rows)


def _y_train_series(n: int = 50, seed: int = 42) -> pd.Series:
    """Builds a synthetic training y series for MASE scaling."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(size=n))


def test_compute_per_fold_metrics_returns_expected_columns():
    """The per-fold DataFrame has exactly the eight documented columns."""
    df = _synthetic_predictions_df()
    y_train = _y_train_series()
    result = compute_per_fold_metrics(df, y_train)
    assert set(result.columns) == {
        "model",
        "scheme",
        "fold_idx",
        "n_observations",
        "rmse",
        "mae",
        "mase",
        "r2",
    }


def test_compute_per_fold_metrics_one_row_per_model_scheme_fold():
    """For 4 models, 2 schemes, 3 folds, the per-fold DataFrame has 24 rows."""
    df = _synthetic_predictions_df(n_models=4, n_schemes=2, fold_sizes=[4, 4, 4])
    y_train = _y_train_series()
    result = compute_per_fold_metrics(df, y_train)
    assert len(result) == 4 * 2 * 3


def test_compute_per_fold_metrics_values_match_compute_all_metrics():
    """For one (model, scheme, fold) group, metrics match an independent compute_all_metrics call."""
    df = _synthetic_predictions_df()
    y_train = _y_train_series()
    result = compute_per_fold_metrics(df, y_train)

    row = result.iloc[0]
    subset = df[
        (df["model"] == row["model"])
        & (df["scheme"] == row["scheme"])
        & (df["fold_idx"] == row["fold_idx"])
    ]
    expected = compute_all_metrics(
        subset["y_true"].to_numpy(), subset["y_pred"].to_numpy(), y_train
    )
    assert row["rmse"] == pytest.approx(expected["rmse"])
    assert row["mae"] == pytest.approx(expected["mae"])
    assert row["mase"] == pytest.approx(expected["mase"])
    assert row["r2"] == pytest.approx(expected["r2"])


def test_aggregate_cv_results_returns_expected_columns():
    """The aggregate DataFrame has exactly the 15 documented columns."""
    df = _synthetic_predictions_df()
    y_train = _y_train_series()
    per_fold = compute_per_fold_metrics(df, y_train)
    agg = aggregate_cv_results(per_fold)
    assert set(agg.columns) == {
        "model",
        "scheme",
        "n_folds",
        "mean_rmse",
        "std_rmse",
        "median_rmse",
        "mean_mae",
        "std_mae",
        "median_mae",
        "mean_mase",
        "std_mase",
        "median_mase",
        "mean_r2",
        "std_r2",
        "median_r2",
    }


def test_aggregate_cv_results_one_row_per_model_scheme():
    """For 4 models and 2 schemes, the aggregate has 8 rows."""
    df = _synthetic_predictions_df(n_models=4, n_schemes=2)
    y_train = _y_train_series()
    per_fold = compute_per_fold_metrics(df, y_train)
    agg = aggregate_cv_results(per_fold)
    assert len(agg) == 8


def test_aggregate_cv_results_mean_matches_numpy_mean():
    """For one (model, scheme), mean_rmse equals numpy.mean of the per-fold rmse values."""
    df = _synthetic_predictions_df()
    y_train = _y_train_series()
    per_fold = compute_per_fold_metrics(df, y_train)
    agg = aggregate_cv_results(per_fold)

    row = agg.iloc[0]
    fold_values = per_fold[
        (per_fold["model"] == row["model"]) & (per_fold["scheme"] == row["scheme"])
    ]["rmse"].to_numpy()
    assert row["mean_rmse"] == pytest.approx(float(np.mean(fold_values)))


def test_aggregate_cv_results_median_matches_numpy_median():
    """median_rmse equals numpy.median of the per-fold rmse values."""
    df = _synthetic_predictions_df()
    y_train = _y_train_series()
    per_fold = compute_per_fold_metrics(df, y_train)
    agg = aggregate_cv_results(per_fold)

    row = agg.iloc[0]
    fold_values = per_fold[
        (per_fold["model"] == row["model"]) & (per_fold["scheme"] == row["scheme"])
    ]["rmse"].to_numpy()
    assert row["median_rmse"] == pytest.approx(float(np.median(fold_values)))


def test_aggregate_cv_results_std_matches_numpy_std_ddof_1():
    """std_rmse equals numpy.std with ddof=1 of the per-fold rmse values."""
    df = _synthetic_predictions_df()
    y_train = _y_train_series()
    per_fold = compute_per_fold_metrics(df, y_train)
    agg = aggregate_cv_results(per_fold)

    row = agg.iloc[0]
    fold_values = per_fold[
        (per_fold["model"] == row["model"]) & (per_fold["scheme"] == row["scheme"])
    ]["rmse"].to_numpy()
    assert row["std_rmse"] == pytest.approx(float(np.std(fold_values, ddof=1)))


def test_aggregate_cv_results_handles_unequal_fold_sizes():
    """Aggregation runs cleanly with varying fold sizes (mimics regime-aligned scheme)."""
    df = _synthetic_predictions_df(fold_sizes=[10, 6, 4, 8])
    y_train = _y_train_series()
    per_fold = compute_per_fold_metrics(df, y_train)
    agg = aggregate_cv_results(per_fold)
    assert len(agg) == 4 * 2
    assert agg["mean_rmse"].notna().all()


def test_aggregate_cv_results_single_fold_returns_nan_std():
    """With one fold per (model, scheme), std is NaN; mean and median are still valid."""
    df = _synthetic_predictions_df(fold_sizes=[4])
    y_train = _y_train_series()
    per_fold = compute_per_fold_metrics(df, y_train)
    agg = aggregate_cv_results(per_fold)

    assert (agg["n_folds"] == 1).all()
    assert agg["std_rmse"].isna().all()
    assert agg["mean_rmse"].notna().all()
    assert agg["median_rmse"].notna().all()
