"""Refits XGBoost on the frozen 2000-2025 history and forecasts 2026 Q1.

X[t] predicts y[t+1] throughout this project, so the 2026 Q1 forecast comes
from the 2025 Q4 row already sitting in the frozen dataset, not from any
newly downloaded 2026 predictor data. Training stays fixed at 2000-2025.

Run with
    PYTHONPATH=. python scripts/forecast_2026_q1.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from loguru import logger

from src.logging_setup import configure_logging
from src.models.train_all import (
    _DATA_PATH,
    _dataset_hash,
    _library_versions,
    _load_dataset,
    _prepare_sklearn_Xy,
    _train_xgboost,
)
from src.models.tune import load_tuning_result

_TUNING_CACHE = Path("results/tuning/xgboost_best_params.json")
_MODEL_OUT = Path("results/models/xgboost_2026_q1_refit.joblib")
_META_OUT = Path("results/models/xgboost_2026_q1_refit_meta.json")
_REPORT_OUT = Path("results/2026-q1-forecast.md")

_LAG_FEATURES = ["gdp_growth", "gdp_lag_1", "gdp_lag_4", "gdp_rolling_mean_4q", "gdp_yoy"]


def refit_xgboost() -> tuple[object, dict, str, int]:
    df = _load_dataset()
    df = df.dropna().reset_index(drop=True)
    dataset_hash = _dataset_hash(_DATA_PATH)

    best_params = load_tuning_result(_TUNING_CACHE)
    if best_params is None:
        raise FileNotFoundError(
            f"No cached tuning result at {_TUNING_CACHE}. Run `make tune` first."
        )

    X, y = _prepare_sklearn_Xy(df)
    model = _train_xgboost(X, y, best_params["best_params"])
    return model, best_params["best_params"], dataset_hash, len(X)


def forecast_row() -> tuple[dict, "object"]:
    """Returns the 2025 Q4 row's lag features and the row itself, ready for predict()."""
    df = _load_dataset()
    df = df.dropna().reset_index(drop=True)
    last_row = df.iloc[[-1]]
    lag_values = {col: float(last_row[col].iloc[0]) for col in _LAG_FEATURES}
    X_forecast = last_row.drop(columns=["date", "regime"])
    return lag_values, X_forecast


def _persist_model(model: object, best_params: dict, dataset_hash: str, n_rows: int) -> None:
    _MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, _MODEL_OUT)
    meta = {
        "model": "xgboost",
        "purpose": "2026 forecast validation epic, Q1 only",
        "best_params": best_params,
        "library_versions": _library_versions(),
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_hash_md5": dataset_hash,
        "n_training_rows": n_rows,
        "training_window": "2000 Q1 to 2025 Q4, frozen, no 2026 data included",
        "random_state": 42,
    }
    _META_OUT.write_text(json.dumps(meta, indent=2))


def _render_report(lag_values: dict, forecast: float, dataset_hash: str, n_rows: int) -> str:
    lines = [
        "# 2026 Q1 forecast",
        "",
        "XGBoost refit on the frozen 2000-2025 history (Sprint 3 cached hyperparameters, "
        "not retuned) and forecast from the 2025 Q4 row already in that frozen dataset. "
        "No 2026 data was used as model input; training stayed at 2000-2025.",
        "",
        f"Training rows: {n_rows}. Dataset hash: {dataset_hash}.",
        "",
        "## Lag and rolling features used (2025 Q4 row, frozen vintage)",
        "",
        "| Feature | Value | Source quarter | Vintage note |",
        "|---|---|---|---|",
        f"| gdp_growth (contemporaneous) | {lag_values['gdp_growth']} | 2025 Q4 | unrevised |",
        f"| gdp_lag_1 | {lag_values['gdp_lag_1']} | 2025 Q3 | unrevised |",
        f"| gdp_lag_4 | {lag_values['gdp_lag_4']} | 2024 Q4 | frozen 0.3, raw now revised to 0.4 |",
        f"| gdp_rolling_mean_4q | {lag_values['gdp_rolling_mean_4q']} | 2025 Q1 to Q4 | window includes 2025 Q1, frozen 0.7, raw now revised to 0.6 |",
        f"| gdp_yoy | {lag_values['gdp_yoy']} | 2025 Q1 to Q4 | same window, same drift note |",
        "",
        "All five values were read directly from data/processed/final_dataset.parquet, "
        "so the drift-affected quarters (2024 Q4 and 2025 Q1) are the frozen training-vintage "
        "values, not the freshly re-downloaded raw ones.",
        "",
        "## Forecast",
        "",
        f"**2026 Q1 gdp_growth forecast: {forecast:.4f}%**",
        "",
        "Not yet validated against the ONS actual (0.6%). That is the next step.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    configure_logging()
    model, best_params, dataset_hash, n_rows = refit_xgboost()
    _persist_model(model, best_params, dataset_hash, n_rows)

    lag_values, X_forecast = forecast_row()
    forecast = float(model.predict(X_forecast)[0])

    report = _render_report(lag_values, forecast, dataset_hash, n_rows)
    _REPORT_OUT.write_text(report)

    logger.info("Refit model saved: {}", _MODEL_OUT)
    logger.info("Forecast report saved: {}", _REPORT_OUT)
    logger.success("2026 Q1 forecast: {:.4f}%", forecast)


if __name__ == "__main__":
    main()
