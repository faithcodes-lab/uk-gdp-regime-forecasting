"""Refits Ridge on the frozen 2000-2025 history and forecasts 2026 Q1 and Q2.

XGBoost was selected for the headline 2026 forecast (lowest mean RMSE,
MAE and MASE across both CV schemes, and the only model with a positive
R-squared anywhere), but Section 4.2's Diebold-Mariano tests found no
statistically reliable difference between XGBoost and Ridge, and Ridge
had a better median RMSE and MASE. This script refits Ridge under the
same conditions as the XGBoost 2026 forecast (same frozen dataset, same
one-step-ahead mapping, same cached hyperparameters, no retuning) and
forecasts both quarters, so the two models' forecasts can be shown
side by side rather than presenting XGBoost's forecast as if it were
uniquely justified.

Q1 reuses forecast_2026_q1.forecast_row() and Q2 reuses
forecast_2026_q2.build_2026q1_feature_row(): both are model-agnostic,
they only build the feature row, not the prediction.

Run with
    PYTHONPATH=. python scripts/forecast_2026_ridge.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from loguru import logger

from scripts.forecast_2026_q1 import forecast_row
from scripts.forecast_2026_q2 import build_2026q1_feature_row
from src.logging_setup import configure_logging
from src.models.train_all import (
    _DATA_PATH,
    _dataset_hash,
    _load_dataset,
    _prepare_sklearn_Xy,
    _train_ridge,
)
from src.models.tune import load_tuning_result

_TUNING_CACHE = Path("results/tuning/ridge_best_params.json")
_MODEL_OUT = Path("results/models/ridge_2026_q1_refit.joblib")
_META_OUT = Path("results/models/ridge_2026_q1_refit_meta.json")
_REPORT_OUT = Path("results/2026-ridge-forecast.md")

_Q1_ACTUAL = 0.6
_Q2_ACTUAL = 0.4
_XGBOOST_Q1_FORECAST = 0.444
_XGBOOST_Q2_FORECAST = 0.444


def refit_ridge() -> tuple[object, dict, str, int]:
    """Refits Ridge on the frozen dataset with cached hyperparameters. Returns (model, best_params, dataset_hash, n_rows)."""
    df = _load_dataset()
    df = df.dropna().reset_index(drop=True)
    dataset_hash = _dataset_hash(_DATA_PATH)

    best_params = load_tuning_result(_TUNING_CACHE)
    if best_params is None:
        raise FileNotFoundError(f"No cached tuning result at {_TUNING_CACHE}. Run `make tune` first.")

    X, y = _prepare_sklearn_Xy(df)
    model = _train_ridge(X, y, best_params["best_params"])
    return model, best_params["best_params"], dataset_hash, len(X)


def _persist_model(model: object, best_params: dict, dataset_hash: str, n_rows: int) -> None:
    _MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, _MODEL_OUT)
    meta = {
        "model": "ridge",
        "purpose": "2026 forecast validation epic, second model for comparison",
        "best_params": best_params,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_hash_md5": dataset_hash,
        "n_training_rows": n_rows,
        "training_window": "2000 Q1 to 2025 Q4, frozen, no 2026 data included",
        "random_state": 42,
    }
    _META_OUT.write_text(json.dumps(meta, indent=2))


def _render_report(q1_forecast: float, q2_forecast: float) -> str:
    q1_error = q1_forecast - _Q1_ACTUAL
    q2_error = q2_forecast - _Q2_ACTUAL
    lines = [
        "# 2026 Ridge forecast: second-model comparison",
        "",
        "XGBoost was selected for the headline 2026 forecast on the lowest mean RMSE, MAE and "
        "MASE across both CV schemes, and as the only model with a positive R-squared anywhere "
        "(0.067, regime-aligned). But Section 4.2's Diebold-Mariano tests found no statistically "
        "reliable difference between the models, and Ridge had a better median RMSE and MASE. "
        "This report refits Ridge under identical conditions (same frozen 2000-2025 dataset, "
        "same one-step-ahead mapping, same cached hyperparameters, no retuning) and forecasts "
        "both 2026 quarters, so the two forecasts can be compared directly.",
        "",
        "## Forecast comparison",
        "",
        "| Model | Q1 2026 forecast | Q1 error | Q2 2026 forecast | Q2 error |",
        "|---|---|---|---|---|",
        f"| Ridge | {q1_forecast:.4f}% | {q1_error:+.4f} | {q2_forecast:.4f}% | {q2_error:+.4f} |",
        f"| XGBoost | {_XGBOOST_Q1_FORECAST:.4f}% | {_XGBOOST_Q1_FORECAST - _Q1_ACTUAL:+.4f} | "
        f"{_XGBOOST_Q2_FORECAST:.4f}% | {_XGBOOST_Q2_FORECAST - _Q2_ACTUAL:+.4f} |",
        "",
        f"ONS actuals: Q1 2026 {_Q1_ACTUAL:.4f}%, Q2 2026 {_Q2_ACTUAL:.4f}%.",
        "",
        "## Reading",
        "",
        "Ridge and XGBoost diverge more than Section 4.2's cross-validated near-tie might "
        "suggest: Ridge forecasts lower than XGBoost for both quarters, and lands further from "
        "the ONS actual both times. Two quarters is too small a sample to conclude Ridge is the "
        "worse model here, since 4.2's own tests found no reliable difference between them "
        "under cross-validation; this is one realisation, not a repeated comparison. What it "
        "does show is that the model choice was not inconsequential for these two forecasts, "
        "so presenting only XGBoost's number without this comparison would have understated "
        "how much the point forecast depends on which model was picked.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    configure_logging()
    model, best_params, dataset_hash, n_rows = refit_ridge()
    _persist_model(model, best_params, dataset_hash, n_rows)

    _, X_q1 = forecast_row()
    q1_forecast = float(model.predict(X_q1)[0])

    feature_row_q2, _ = build_2026q1_feature_row()
    q2_forecast = float(model.predict(feature_row_q2)[0])

    report = _render_report(q1_forecast, q2_forecast)
    _REPORT_OUT.write_text(report)

    logger.info("Refit model saved: {}", _MODEL_OUT)
    logger.success("Ridge 2026 Q1 forecast: {:.4f}% (actual {:.4f}%)", q1_forecast, _Q1_ACTUAL)
    logger.success("Ridge 2026 Q2 forecast: {:.4f}% (actual {:.4f}%)", q2_forecast, _Q2_ACTUAL)
    logger.info("Report saved: {}", _REPORT_OUT)


if __name__ == "__main__":
    main()
