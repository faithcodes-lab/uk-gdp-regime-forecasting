"""Forecasts 2026 Q2 from the constructed 2026 Q1 feature row.

X[t] predicts y[t+1], so the 2026 Q2 forecast comes from a 2026 Q1 feature
row that does not exist in the frozen dataset and has to be built: the real
2026 Q1 predictors and the published 2026 Q1 growth actual (0.6 percent),
combined with GDP and confidence history taken from the frozen dataset so
the lag and rolling features stay on the vintage the model was trained on.
The row is assembled by appending a 2026 Q1 row to the frozen raw columns
and running the pipeline's own engineer_features, so the engineered
features are computed exactly as in training. Training stays fixed at
2000-2025.

Run with
    PYTHONPATH=. python scripts/forecast_2026_q2.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.data.build_dataset import aggregate_to_quarterly, load_raw_from_disk
from src.features.engineer import engineer_features
from src.logging_setup import configure_logging
from src.models.train_all import _DATA_PATH, _prepare_sklearn_Xy

_REFIT_MODEL = Path("results/models/xgboost_2026_q1_refit.joblib")
_REPORT_OUT = Path("results/2026-q2-forecast.md")
_Q1_2026_ACTUAL = 0.6
_Q1_2026_DATE = pd.Timestamp("2026-03-31")

_RAW_PREDICTORS = [
    "unemployment_rate",
    "cpi_inflation",
    "trade_balance",
    "gfcf_growth",
    "govt_consumption_growth",
    "bank_rate",
    "gbp_usd_rate",
    "brent_oil",
    "business_confidence",
    "consumer_confidence",
]
_GILTS = ["gilt_2y_yield", "gilt_10y_yield"]


def build_2026q1_feature_row() -> tuple[pd.DataFrame, pd.Series]:
    """Builds the 2026 Q1 feature row (which predicts 2026 Q2) and returns it with the training columns."""
    frozen = pd.read_parquet(_DATA_PATH)
    training_cols = list(_prepare_sklearn_Xy(frozen.dropna().reset_index(drop=True))[0].columns)

    raw_2026q1 = aggregate_to_quarterly(load_raw_from_disk())
    q1 = raw_2026q1.loc[raw_2026q1["date"] == _Q1_2026_DATE]
    if q1.empty:
        raise ValueError("No aggregated 2026 Q1 row found; run `make download` first.")

    # Frozen raw columns carry the training-vintage GDP and confidence history; the
    # two gilt columns are not kept in the frozen dataset, so they are NaN for the
    # historical rows and only need a real value on the 2026 Q1 row (yield_curve_slope
    # is a contemporaneous subtraction, so the history is never used for it).
    base = frozen[["date", "gdp_growth", *_RAW_PREDICTORS]].copy()
    base[_GILTS[0]] = np.nan
    base[_GILTS[1]] = np.nan

    new_row = {"date": _Q1_2026_DATE, "gdp_growth": _Q1_2026_ACTUAL}
    for col in _RAW_PREDICTORS + _GILTS:
        new_row[col] = float(q1[col].iloc[0])

    combined = pd.concat([base, pd.DataFrame([new_row])], ignore_index=True)
    engineered = engineer_features(combined)
    row = engineered.loc[engineered["date"] == _Q1_2026_DATE]
    return row[training_cols].reset_index(drop=True), pd.Series(training_cols)


def _render_report(feature_row: pd.DataFrame, forecast: float) -> str:
    r = feature_row.iloc[0]
    lines = [
        "# 2026 Q2 forecast",
        "",
        "XGBoost refit on the frozen 2000-2025 history (Sprint 3 cached hyperparameters, "
        "not retuned) and forecast from a constructed 2026 Q1 feature row. Under the "
        "one-step-ahead mapping the 2026 Q1 row predicts 2026 Q2. Training stayed at "
        "2000-2025; the 2026 data is predictor input only.",
        "",
        "## How the 2026 Q1 feature row was built",
        "",
        "The GDP and confidence history that feeds the lag and rolling features was taken "
        "from the frozen dataset (training vintage), the real 2026 Q1 predictors were "
        "aggregated from the refreshed raw data, and the contemporaneous gdp_growth is the "
        "published 2026 Q1 actual (0.6 percent). The row was assembled by appending a 2026 Q1 "
        "row to the frozen raw columns and running the pipeline's engineer_features, so the "
        "engineered features match training.",
        "",
        "## GDP-history features on the 2026 Q1 row (frozen vintage)",
        "",
        "| Feature | Value | Source quarter | Vintage note |",
        "|---|---|---|---|",
        f"| gdp_growth (contemporaneous) | {r['gdp_growth']:.4f} | 2026 Q1 | published actual (0.6) |",
        f"| gdp_lag_1 | {r['gdp_lag_1']:.4f} | 2025 Q4 | frozen |",
        f"| gdp_lag_4 | {r['gdp_lag_4']:.4f} | 2025 Q1 | frozen 0.7, raw now revised to 0.6 |",
        f"| gdp_rolling_mean_4q | {r['gdp_rolling_mean_4q']:.4f} | 2025 Q2 to 2026 Q1 | frozen 2025, new 2026 Q1 |",
        f"| gdp_yoy | {r['gdp_yoy']:.4f} | 2025 Q2 to 2026 Q1 | frozen 2025, new 2026 Q1 |",
        "",
        "## Real 2026 Q1 predictors and derived features",
        "",
        "| Feature | Value |",
        "|---|---|",
    ]
    for col in _RAW_PREDICTORS + ["business_confidence_rolling_mean_4q", "yield_curve_slope"]:
        lines.append(f"| {col} | {r[col]:.4f} |")
    lines += [
        "",
        "## Forecast",
        "",
        f"**2026 Q2 gdp_growth forecast: {forecast:.4f}%**",
        "",
        "Not validated: ONS has not published 2026 Q2 GDP yet (next release 13 August 2026). "
        "The forecast will be validated against the published actual then, the same way as Q1.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    configure_logging()
    feature_row, _ = build_2026q1_feature_row()
    model = joblib.load(_REFIT_MODEL)
    forecast = float(model.predict(feature_row)[0])
    _REPORT_OUT.write_text(_render_report(feature_row, forecast))
    logger.info("feature row (2026 Q1, predicts 2026 Q2):")
    for col, val in feature_row.iloc[0].items():
        logger.info("  {:35s} {:.6f}", col, val)
    logger.success("2026 Q2 forecast: {:.4f}%", forecast)
    logger.info("report saved: {}", _REPORT_OUT)


if __name__ == "__main__":
    main()
