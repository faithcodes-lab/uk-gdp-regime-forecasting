"""Why does XGBoost forecast the same value for 2026 Q1 and Q2?

Confirmatory diagnostic, not a model change. Checks two explanations
for the identical Q1/Q2 forecast directly, rather than asserting one:
whether the two rows land in the same terminal leaf despite different
inputs, and whether the model behaves close to a constant-mean
predictor on ordinary (non-crisis) quarters generally.

Run with
    PYTHONPATH=. python scripts/near_constant_prediction_check.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from scripts.forecast_2026_q1 import forecast_row
from scripts.forecast_2026_q2 import build_2026q1_feature_row
from src.logging_setup import configure_logging
from src.models.train_all import _load_dataset, _prepare_sklearn_Xy

_REFIT_MODEL = Path("results/models/xgboost_2026_q1_refit.joblib")
_REPORT_OUT = Path("results/near-constant-prediction-check.md")
_CRISIS_REGIMES = ["Global Financial Crisis", "COVID-19 Shock"]


def same_terminal_leaves(model, X_a: pd.DataFrame, X_b: pd.DataFrame) -> bool:
    """Returns True if two single-row feature matrices land in the same leaf in every tree."""
    leaves_a = model.get_estimator().apply(X_a)
    leaves_b = model.get_estimator().apply(X_b)
    return bool(np.array_equal(leaves_a, leaves_b))


def constant_mean_comparison(y_true: np.ndarray, y_pred: np.ndarray, train_mean: float) -> dict:
    """Returns prediction-concentration and constant-mean-benchmark stats for a subset of rows."""
    rounded = np.round(y_pred, 4)
    counts = pd.Series(rounded).value_counts()
    constant_rmse = float(np.sqrt(np.mean((y_true - train_mean) ** 2)))
    model_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {
        "n": len(y_pred),
        "n_distinct_predictions": int(counts.shape[0]),
        "most_common_count": int(counts.iloc[0]),
        "pred_std": float(np.std(y_pred)),
        "actual_std": float(np.std(y_true)),
        "constant_mean_rmse": constant_rmse,
        "model_rmse": model_rmse,
        "improvement_pct": 100 * (constant_rmse - model_rmse) / constant_rmse,
    }


def run_check() -> tuple[bool, dict]:
    """Returns (same_leaves_for_q1_q2, constant_mean_comparison_on_normal_quarters)."""
    model = joblib.load(_REFIT_MODEL)
    _, X_q1 = forecast_row()
    X_q2, _ = build_2026q1_feature_row()
    same_leaves = same_terminal_leaves(model, X_q1, X_q2)

    df = _load_dataset().dropna().reset_index(drop=True)
    X, y = _prepare_sklearn_Xy(df)
    regime = df["regime"].iloc[:-1].reset_index(drop=True)
    normal_mask = ~regime.isin(_CRISIS_REGIMES)

    preds = model.predict(X)
    stats = constant_mean_comparison(
        y[normal_mask].to_numpy(), preds[normal_mask], float(y.mean())
    )
    return same_leaves, stats


def _write_report(same_leaves: bool, stats: dict, out_path: Path = _REPORT_OUT) -> None:
    lines = [
        "# Why does XGBoost forecast the same value for 2026 Q1 and Q2?",
        "",
        "Confirmatory diagnostic, not a model change.",
        "",
        f"**Same terminal leaf in every tree for the Q1 and Q2 rows despite different "
        f"inputs: {same_leaves}.**",
        "",
        f"Across {stats['n']} non-crisis quarters, the model predicts "
        f"{stats['n_distinct_predictions']} distinct values, the most common one "
        f"{stats['most_common_count']} times "
        f"({100 * stats['most_common_count'] / stats['n']:.0f}%).",
        "",
        f"Prediction std: {stats['pred_std']:.4f}. Actual std: {stats['actual_std']:.4f}.",
        "",
        f"Constant-mean benchmark RMSE: {stats['constant_mean_rmse']:.4f}. "
        f"Model RMSE: {stats['model_rmse']:.4f} "
        f"({stats['improvement_pct']:.1f}% improvement over always predicting the mean).",
        "",
    ]
    out_path.write_text("\n".join(lines))
    logger.success("wrote {}", out_path)


def main() -> None:
    configure_logging()
    same_leaves, stats = run_check()
    logger.info("same terminal leaves for Q1 and Q2: {}", same_leaves)
    logger.info(
        "{} distinct predictions across {} non-crisis quarters, most common used {} times",
        stats["n_distinct_predictions"],
        stats["n"],
        stats["most_common_count"],
    )
    logger.info(
        "constant-mean RMSE={:.4f}, model RMSE={:.4f} ({:.1f}% improvement)",
        stats["constant_mean_rmse"],
        stats["model_rmse"],
        stats["improvement_pct"],
    )
    _write_report(same_leaves, stats)


if __name__ == "__main__":
    main()
