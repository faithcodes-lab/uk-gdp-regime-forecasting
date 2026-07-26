"""Per-fold and across-fold aggregation of forecast accuracy metrics.

Two functions: compute_per_fold_metrics groups CP2's predictions by
(model, scheme, fold_idx) and computes RMSE, MAE, MASE, R squared per
group; aggregate_cv_results then groups the per-fold metrics by
(model, scheme) and reports mean, median, and std for each metric.
Aggregation is unweighted across folds, with n_observations preserved
on the per-fold DataFrame so weighted statistics or the CV-variance
boxplot can use it directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_mae, compute_mase, compute_r2, compute_rmse


def compute_per_fold_metrics(
    predictions_df: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """Returns one row per (model, scheme, fold_idx) with the four metrics.

    MASE uses the full y_train series so values are comparable across
    folds, schemes, and (jointly with CP4) across regimes. Empty groups
    are silently skipped.
    """
    rows: list[dict[str, Any]] = []
    for (model_name, scheme, fold_idx), group in predictions_df.groupby(
        ["model", "scheme", "fold_idx"]
    ):
        n = len(group)
        if n == 0:
            continue
        y_true = group["y_true"].to_numpy()
        y_pred = group["y_pred"].to_numpy()
        rows.append(
            {
                "model": model_name,
                "scheme": scheme,
                "fold_idx": int(fold_idx),
                "n_observations": n,
                "rmse": compute_rmse(y_true, y_pred),
                "mae": compute_mae(y_true, y_pred),
                "mase": compute_mase(y_true, y_pred, y_train),
                "r2": compute_r2(y_true, y_pred),
            }
        )
    return pd.DataFrame(rows)


def aggregate_cv_results(per_fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregates per-fold metrics into mean, median, and std per (model, scheme).

    Uses ddof=1 (sample std), so a single-fold group yields NaN std.
    Aggregation is unweighted across folds.
    """
    rows: list[dict[str, Any]] = []
    for (model_name, scheme), group in per_fold_metrics.groupby(["model", "scheme"]):
        n_folds = len(group)
        if n_folds == 0:
            continue
        row: dict[str, Any] = {
            "model": model_name,
            "scheme": scheme,
            "n_folds": n_folds,
        }
        for metric in ("rmse", "mae", "mase", "r2"):
            values = group[metric].to_numpy()
            row[f"mean_{metric}"] = float(np.mean(values))
            # ddof=1 with N=1 is undefined; report NaN explicitly to avoid the
            # numpy runtime warning that would otherwise leak into test output
            row[f"std_{metric}"] = float(np.std(values, ddof=1)) if n_folds > 1 else float("nan")
            row[f"median_{metric}"] = float(np.median(values))
        rows.append(row)
    return pd.DataFrame(rows)
