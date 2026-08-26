"""How sensitive are the crisis-regime results to individual quarters?

Confirmatory diagnostic, not a model change. Global Financial Crisis and
COVID-19 Shock each have only 6 quarters under regime-aligned CV, so a
single unusual quarter could be driving the whole regime's headline
RMSE. This computes leave-one-quarter-out RMSE for each crisis regime,
model, and held-out quarter: refits nothing, just recomputes RMSE on
the regime's existing regime-aligned predictions with one quarter
dropped at a time, and reports how far each drop moves the metric from
the full-regime value.

Run with
    PYTHONPATH=. python scripts/regime_imbalance_sensitivity.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from src.evaluation.metrics import compute_rmse
from src.logging_setup import configure_logging

_PREDICTIONS_PATH = Path("results/predictions/predictions.parquet")
_REPORT_OUT = Path("results/regime-imbalance-sensitivity.md")
_CRISIS_REGIMES = ["Global Financial Crisis", "COVID-19 Shock"]
_SCHEME = "regime_aligned"


def leave_one_quarter_out_rmse(group: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per held-out quarter: RMSE on the remaining quarters, and the swing from the full-regime RMSE."""
    full_rmse = compute_rmse(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
    rows = []
    for held_out_quarter in group["quarter"].unique():
        remaining = group[group["quarter"] != held_out_quarter]
        loo_rmse = compute_rmse(remaining["y_true"].to_numpy(), remaining["y_pred"].to_numpy())
        swing = loo_rmse - full_rmse
        swing_pct = 100 * swing / full_rmse if full_rmse != 0 else 0.0
        rows.append(
            {
                "held_out_quarter": held_out_quarter,
                "full_regime_rmse": full_rmse,
                "leave_one_out_rmse": loo_rmse,
                "swing": swing,
                "swing_pct": swing_pct,
            }
        )
    return pd.DataFrame(rows)


def run_sensitivity_check(predictions_path: Path | str = _PREDICTIONS_PATH) -> pd.DataFrame:
    """Returns one row per (model, regime, held-out quarter) for the two crisis regimes under regime-aligned CV."""
    preds = pd.read_parquet(predictions_path)
    subset = preds[(preds["scheme"] == _SCHEME) & (preds["regime"].isin(_CRISIS_REGIMES))]

    all_rows = []
    for (model_name, regime), group in subset.groupby(["model", "regime"]):
        loo = leave_one_quarter_out_rmse(group)
        loo.insert(0, "regime", regime)
        loo.insert(0, "model", model_name)
        all_rows.append(loo)
    return pd.concat(all_rows, ignore_index=True)


def _write_report(results: pd.DataFrame, out_path: Path = _REPORT_OUT) -> None:
    worst = results.loc[results["swing_pct"].abs().idxmax()]
    max_swing_by_regime = results.groupby("regime")["swing_pct"].apply(lambda s: s.abs().max())

    lines = [
        "# How sensitive are the crisis-regime results to individual quarters?",
        "",
        "Confirmatory diagnostic, not a model change. Leave-one-quarter-out RMSE for each "
        "crisis regime (Global Financial Crisis, COVID-19 Shock), model, and held-out quarter "
        "under regime-aligned CV, each with only 6 quarters.",
        "",
        "## Largest swing per regime",
        "",
        "| Regime | Largest |swing| across all models and quarters |",
        "|---|---|",
    ]
    for regime, swing in max_swing_by_regime.items():
        lines.append(f"| {regime} | {swing:.1f}% |")

    lines += [
        "",
        f"Single largest swing overall: {worst['model']} in {worst['regime']}, dropping "
        f"{pd.Timestamp(worst['held_out_quarter']).date()} "
        f"moves RMSE from {worst['full_regime_rmse']:.3f} to {worst['leave_one_out_rmse']:.3f} "
        f"({worst['swing_pct']:+.1f}%).",
        "",
        "## Full results",
        "",
        "| Model | Regime | Held-out quarter | Full RMSE | Leave-one-out RMSE | Swing |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in results.iterrows():
        lines.append(
            f"| {r['model']} | {r['regime']} | {pd.Timestamp(r['held_out_quarter']).date()} | "
            f"{r['full_regime_rmse']:.3f} | {r['leave_one_out_rmse']:.3f} | {r['swing_pct']:+.1f}% |"
        )
    lines.append("")
    out_path.write_text("\n".join(lines))
    logger.success("wrote {}", out_path)


def main() -> None:
    configure_logging()
    results = run_sensitivity_check()
    max_swing_by_regime = results.groupby("regime")["swing_pct"].apply(lambda s: s.abs().max())
    for regime, swing in max_swing_by_regime.items():
        logger.info("{}: largest single-quarter swing = {:.1f}%", regime, swing)
    _write_report(results)


if __name__ == "__main__":
    main()
