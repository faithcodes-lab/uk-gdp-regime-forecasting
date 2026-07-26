"""Validates the 2026 Q1 forecast against the published ONS actual.

Compares the forecast in results/2026-q1-forecast.md against the ONS Q1
2026 figure, and sets the error against XGBoost's usual error in the
Post-COVID Recovery regime from the Sprint 4 evaluation, since Q1 2026
falls in that regime.

Run with
    PYTHONPATH=. python scripts/validate_2026_q1.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from scripts.forecast_2026_q1 import forecast_row, refit_xgboost
from src.logging_setup import configure_logging

_ONS_RAW = Path("data/raw/ons/gdp_growth.csv")
_PER_REGIME_METRICS = Path("results/metrics/per_regime.csv")
_REPORT_OUT = Path("results/2026-q1-validation.md")

_ACTUAL_QUARTER = "2026-03-31"
_REGIME = "Post-COVID Recovery"


def _load_actual() -> float:
    df = pd.read_csv(_ONS_RAW, parse_dates=["date"])
    row = df.loc[df["date"] == pd.Timestamp(_ACTUAL_QUARTER)]
    if row.empty:
        raise ValueError(f"No ONS actual found for {_ACTUAL_QUARTER} in {_ONS_RAW}")
    return float(row["value"].iloc[0])


def _load_regime_context() -> pd.DataFrame:
    df = pd.read_csv(_PER_REGIME_METRICS)
    return df.loc[(df["model"] == "xgboost") & (df["regime"] == _REGIME)]


def _render_report(forecast: float, actual: float, regime_rows: pd.DataFrame) -> str:
    error = forecast - actual
    abs_error = abs(error)
    lines = [
        "# 2026 Q1 validation",
        "",
        f"Forecast: {forecast:.4f}%. ONS actual (2026 Q1, real, CVM, SA, QoQ): {actual:.4f}%.",
        f"Error: {error:+.4f} percentage points (absolute error {abs_error:.4f}).",
        "",
        f"## Context: XGBoost's usual error in {_REGIME}",
        "",
        "Q1 2026 falls in the Post-COVID Recovery regime, so this single forecast is read "
        "against XGBoost's typical error there in the Sprint 4 evaluation, rather than "
        "judged on its own.",
        "",
        "| Scheme | n | RMSE | MAE | MASE | R2 |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in regime_rows.iterrows():
        lines.append(
            f"| {r['scheme']} | {int(r['n_observations'])} | {r['rmse']:.3f} | "
            f"{r['mae']:.3f} | {r['mase']:.3f} | {r['r2']:.3f} |"
        )
    expanding = regime_rows.loc[regime_rows["scheme"] == "expanding_window"]
    if not expanding.empty:
        mae = float(expanding["mae"].iloc[0])
        rmse = float(expanding["rmse"].iloc[0])
        lines += [
            "",
            f"The Q1 2026 absolute error ({abs_error:.4f}) is "
            f"{abs_error / mae:.2f} times the expanding-window MAE for this regime "
            f"({mae:.3f}) and {abs_error / rmse:.2f} times its RMSE ({rmse:.3f}). "
            "A single quarter cannot confirm a trend either way; this is one data "
            "point read against the regime's usual error range, not a claim that "
            "the model has improved or degraded.",
        ]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    configure_logging()
    model, _, _, _ = refit_xgboost()
    _, X_forecast = forecast_row()
    forecast = float(model.predict(X_forecast)[0])

    actual = _load_actual()
    regime_rows = _load_regime_context()
    report = _render_report(forecast, actual, regime_rows)
    _REPORT_OUT.write_text(report)
    logger.success(
        "Q1 2026: forecast {:.4f}%, actual {:.4f}%, error {:+.4f}",
        forecast,
        actual,
        forecast - actual,
    )
    logger.info("Validation report saved: {}", _REPORT_OUT)


if __name__ == "__main__":
    main()
