"""Rolling-window check: is the near-null finding a side effect of expanding-window CV?

Confirmatory diagnostic, not a model change. Every other result in this
project uses expanding-window CV, where the training set grows over
time. If XGBoost's near-null forecasting result were an artefact of
that growing window rather than a property of the data, a fixed-size
rolling window (same test folds, same cached hyperparameters, but a
training set that slides forward instead of accumulating) should give
a different answer.

No committed script or results file previously existed for this test.
The dissertation text asserts the rolling-window scheme "provides the same results as
the expanding-window scheme," but that specific comparison could not be
independently verified before this script existed. This script
reconstructs the methodology (rolling_window_splits in src/models/cv.py,
same window size as expanding-window's first fold, same 8-fold/4-quarter
boundaries so the two schemes are directly comparable) and reports the
actual result rather than assuming the prior claim.

Run with
    PYTHONPATH=. python scripts/rolling_window_check.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.evaluation.metrics import compute_mae, compute_rmse
from src.logging_setup import configure_logging
from src.models.cv import expanding_window_splits, rolling_window_splits
from src.models.tune import load_tuning_result
from src.models.train_all import _DATA_PATH, _prepare_sklearn_Xy
from src.models.xgboost_model import build_xgboost_pipeline

_TUNING_PATH = Path("results/tuning/xgboost_best_params.json")
_REPORT_OUT = Path("results/rolling-window-check.md")
_RANDOM_STATE = 42


def _cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    xgb_params: dict,
) -> tuple[float, float]:
    """Runs CV for XGBoost over the given splits, refitting per fold. Returns (rmse, mae)."""
    fold_rmse: list[float] = []
    fold_mae: list[float] = []
    for train_idx, test_idx in splits:
        pipeline = build_xgboost_pipeline(
            max_depth=xgb_params["xgboost__max_depth"],
            learning_rate=xgb_params["xgboost__learning_rate"],
            n_estimators=xgb_params["xgboost__n_estimators"],
            random_state=_RANDOM_STATE,
        )
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        fold_rmse.append(compute_rmse(y_test.to_numpy(), y_pred))
        fold_mae.append(compute_mae(y_test.to_numpy(), y_pred))
    return float(np.mean(fold_rmse)), float(np.mean(fold_mae))


def run_rolling_window_check(data_path: Path | str = _DATA_PATH) -> dict:
    """Returns a dict with expanding- and rolling-window RMSE/MAE for XGBoost, same test folds."""
    df = pd.read_parquet(data_path)
    X, y = _prepare_sklearn_Xy(df)
    xgb_params = load_tuning_result(_TUNING_PATH)["best_params"]

    expanding_splits = expanding_window_splits(X, n_splits=8, test_size=4)
    rolling_splits = rolling_window_splits(X, n_splits=8, test_size=4)

    exp_rmse, exp_mae = _cross_validate(X, y, expanding_splits, xgb_params)
    roll_rmse, roll_mae = _cross_validate(X, y, rolling_splits, xgb_params)

    return {
        "expanding_rmse": exp_rmse,
        "expanding_mae": exp_mae,
        "rolling_rmse": roll_rmse,
        "rolling_mae": roll_mae,
        "window_size": len(rolling_splits[0][0]),
    }


def _write_report(result: dict, out_path: Path = _REPORT_OUT) -> None:
    rmse_diff = result["rolling_rmse"] - result["expanding_rmse"]
    rmse_diff_pct = 100 * rmse_diff / result["expanding_rmse"]
    close = abs(rmse_diff_pct) < 10

    lines = [
        "# Rolling-window check: is the near-null finding a side effect of expanding-window CV?",
        "",
        "Confirmatory diagnostic, not a model change. Reconstructed methodology (see script "
        "docstring): no committed script, results file, or decision-log entry previously existed "
        "for this test.",
        "",
        f"Fixed training window of {result['window_size']} quarters, sliding forward, with test "
        "fold boundaries identical to the primary expanding-window scheme (8 folds, 4 quarters "
        "each) so the two are directly comparable; same cached XGBoost hyperparameters as the "
        "main evaluation.",
        "",
        "## Accuracy comparison",
        "",
        "| Scheme | RMSE | MAE |",
        "|---|---|---|",
        f"| Expanding-window | {result['expanding_rmse']:.3f} | {result['expanding_mae']:.3f} |",
        f"| Rolling-window | {result['rolling_rmse']:.3f} | {result['rolling_mae']:.3f} |",
        "",
        f"RMSE difference: {rmse_diff:+.3f} ({rmse_diff_pct:+.1f}%).",
        "",
        (
            "The two schemes give a similar result, so the near-null finding does not appear to "
            "be a side effect of the training set growing under the expanding-window scheme."
            if close
            else "The two schemes give a meaningfully different result. The near-null finding "
            "is not confirmed as scheme-independent by this test, and the dissertation text "
            "claiming 'the same results' should be revised to reflect the actual gap measured "
            "here rather than asserted without a supporting script."
        ),
        "",
    ]
    out_path.write_text("\n".join(lines))
    logger.success("wrote {}", out_path)


def main() -> None:
    configure_logging()
    result = run_rolling_window_check()

    logger.info(
        "expanding-window: RMSE={:.3f} MAE={:.3f}", result["expanding_rmse"], result["expanding_mae"]
    )
    logger.info(
        "rolling-window (window={} quarters): RMSE={:.3f} MAE={:.3f}",
        result["window_size"],
        result["rolling_rmse"],
        result["rolling_mae"],
    )
    rmse_diff_pct = (
        100 * (result["rolling_rmse"] - result["expanding_rmse"]) / result["expanding_rmse"]
    )
    logger.info("RMSE difference: {:+.1f}%", rmse_diff_pct)

    _write_report(result)


if __name__ == "__main__":
    main()
