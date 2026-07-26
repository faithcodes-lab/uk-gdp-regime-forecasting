"""Per-regime evaluation of forecast accuracy.

Breaks down model performance across the six economic regimes by grouping
the long-format predictions DataFrame from CP2 and applying CP1's metric
functions. Reports point estimates with a small-sample flag; bootstrap
confidence intervals are exposed as a separate function for the
orchestrator to call on flagged small-sample regimes. IID bootstrap
resampling is used because block bootstrap is impractical at n=6; this
understates uncertainty by ignoring error autocorrelation and is noted
as a methodological limitation in the dissertation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_mae, compute_mase, compute_r2, compute_rmse


def evaluate_per_regime(
    predictions_df: pd.DataFrame,
    y_train: pd.Series,
    small_sample_threshold: int = 10,
) -> pd.DataFrame:
    """Returns one row per (model, scheme, regime) with the four metrics and a small-sample flag.

    MASE is scaled by the full y_train series so the value is comparable
    across all rows of the output, not just within a single (model, scheme).
    Empty (model, scheme, regime) combinations are silently skipped.
    """
    rows: list[dict[str, Any]] = []
    for (model_name, scheme, regime), group in predictions_df.groupby(
        ["model", "scheme", "regime"]
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
                "regime": regime,
                "n_observations": n,
                "rmse": compute_rmse(y_true, y_pred),
                "mae": compute_mae(y_true, y_pred),
                "mase": compute_mase(y_true, y_pred, y_train),
                "r2": compute_r2(y_true, y_pred),
                "small_sample": n < small_sample_threshold,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_regime_metrics(
    predictions_subset: pd.DataFrame,
    y_train: pd.Series,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    confidence_level: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Returns percentile-based bootstrap CIs for each of rmse, mae, mase, R squared.

    IID resampling with replacement over the rows of predictions_subset.
    Each iteration computes all four metrics on the resample; iterations
    that produce a degenerate sample (constant y_true after resample, so
    R squared is undefined) are silently skipped. CI is computed from the
    surviving iterations.
    """
    n = len(predictions_subset)
    if n < 2:
        raise ValueError(f"bootstrap requires at least 2 observations, got {n}")

    rng = np.random.default_rng(random_state)
    y_true_arr = predictions_subset["y_true"].to_numpy()
    y_pred_arr = predictions_subset["y_pred"].to_numpy()
    indices = np.arange(n)

    samples: dict[str, list[float]] = {"rmse": [], "mae": [], "mase": [], "r2": []}
    for _ in range(n_bootstrap):
        sample_idx = rng.choice(indices, size=n, replace=True)
        yt = y_true_arr[sample_idx]
        yp = y_pred_arr[sample_idx]
        try:
            rmse = compute_rmse(yt, yp)
            mae = compute_mae(yt, yp)
            mase = compute_mase(yt, yp, y_train)
            r2 = compute_r2(yt, yp)
        except (ValueError, ZeroDivisionError):
            continue
        # skip degenerate iterations that yielded a non-finite metric (e.g. R squared
        # on a constant y_true resample, where sklearn returns -inf or NaN with a warning)
        if not all(np.isfinite([rmse, mae, mase, r2])):
            continue
        samples["rmse"].append(rmse)
        samples["mae"].append(mae)
        samples["mase"].append(mase)
        samples["r2"].append(r2)

    alpha = (1 - confidence_level) / 2
    cis: dict[str, tuple[float, float]] = {}
    for metric_name, metric_samples in samples.items():
        if not metric_samples:
            cis[metric_name] = (float("nan"), float("nan"))
        else:
            lower = float(np.percentile(metric_samples, 100 * alpha))
            upper = float(np.percentile(metric_samples, 100 * (1 - alpha)))
            cis[metric_name] = (lower, upper)
    return cis
