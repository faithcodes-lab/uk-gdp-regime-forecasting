"""Feature ablation: does removing the macroeconomic predictors cost XGBoost anything?

Refits XGBoost under the same frozen 2000-2025 dataset, the same
expanding-window CV, and the same cached Sprint 3 hyperparameters as the
main evaluation, on three non-overlapping feature sets: the full 17
columns, GDP-history only (5 columns), and macro-only (12 columns). The
full-feature row is expected to reproduce the recorded main-evaluation
RMSE/MAE exactly, which certifies this diagnostic uses the same data,
CV, and model as the headline results (Section 4.5, Test one).

Predictions are saved per observation with their regime label attached,
so a follow-up regime-specific breakdown (e.g. isolating COVID-19 Shock)
can be computed later by filtering the saved parquet rather than rerunning
training.

Run with
    PYTHONPATH=. python scripts/feature_ablation.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.evaluation.metrics import compute_mae, compute_rmse
from src.logging_setup import configure_logging
from src.models.cv import expanding_window_splits
from src.models.tune import load_tuning_result
from src.models.train_all import _DATA_PATH, _prepare_sklearn_Xy
from src.models.xgboost_model import build_xgboost_pipeline

_TUNING_PATH = Path("results/tuning/xgboost_best_params.json")
_PREDICTIONS_OUT = Path("results/predictions/feature_ablation_predictions.parquet")
_REPORT_OUT = Path("results/feature-ablation.md")

_GDP_HISTORY_FEATURES = [
    "gdp_growth",
    "gdp_lag_1",
    "gdp_lag_4",
    "gdp_rolling_mean_4q",
    "gdp_yoy",
]


def _feature_sets(all_columns: list[str]) -> dict[str, list[str]]:
    """Returns the three non-overlapping feature-set variants by column name."""
    macro_only = [c for c in all_columns if c not in _GDP_HISTORY_FEATURES]
    return {
        "full": list(all_columns),
        "gdp_history_only": list(_GDP_HISTORY_FEATURES),
        "macro_only": macro_only,
    }


def _run_variant(
    variant_name: str,
    columns: list[str],
    X: pd.DataFrame,
    y: pd.Series,
    regime: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    xgb_params: dict,
) -> pd.DataFrame:
    """Runs expanding-window CV for one feature-set variant, refitting per fold.

    Returns a long-format DataFrame with one row per (fold, test observation).
    """
    rows: list[dict] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        pipeline = build_xgboost_pipeline(
            max_depth=xgb_params["xgboost__max_depth"],
            learning_rate=xgb_params["xgboost__learning_rate"],
            n_estimators=xgb_params["xgboost__n_estimators"],
            random_state=42,
        )
        X_train, y_train = X.iloc[train_idx][columns], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx][columns], y.iloc[test_idx]
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        for pos, idx in enumerate(test_idx):
            rows.append(
                {
                    "variant": variant_name,
                    "fold_idx": fold_idx,
                    "regime": regime.iloc[idx],
                    "y_true": float(y_test.iloc[pos]),
                    "y_pred": float(y_pred[pos]),
                }
            )
    return pd.DataFrame(rows)


def run_feature_ablation(
    data_path: Path | str = _DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs the three-variant feature ablation and returns (predictions, summary).

    summary has one row per variant with mean RMSE/MAE across the 8
    expanding-window folds (unweighted mean of per-fold metrics, matching
    aggregate_cv_results so the full-variant row is directly comparable to
    the main evaluation's recorded XGBoost RMSE/MAE).
    """
    df = pd.read_parquet(data_path)
    X, y = _prepare_sklearn_Xy(df)
    # target_df aligns row-for-row with X/y (both already one-step shifted and
    # trimmed by _prepare_sklearn_Xy), so regime labels come from the same rows.
    regime = df["regime"].iloc[:-1].reset_index(drop=True)

    splits = expanding_window_splits(X, n_splits=8, test_size=4)
    xgb_params = load_tuning_result(_TUNING_PATH)["best_params"]

    variants = _feature_sets(list(X.columns))
    all_predictions = []
    summary_rows = []
    for variant_name, columns in variants.items():
        logger.info("running variant '{}' ({} features)", variant_name, len(columns))
        preds = _run_variant(variant_name, columns, X, y, regime, splits, xgb_params)
        all_predictions.append(preds)

        per_fold_rmse = preds.groupby("fold_idx").apply(
            lambda g: compute_rmse(g["y_true"].to_numpy(), g["y_pred"].to_numpy())
        )
        per_fold_mae = preds.groupby("fold_idx").apply(
            lambda g: compute_mae(g["y_true"].to_numpy(), g["y_pred"].to_numpy())
        )
        summary_rows.append(
            {
                "variant": variant_name,
                "n_features": len(columns),
                "features": ", ".join(columns),
                "rmse": per_fold_rmse.mean(),
                "mae": per_fold_mae.mean(),
            }
        )

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    return predictions_df, summary_df


def _write_report(summary_df: pd.DataFrame, out_path: Path = _REPORT_OUT) -> None:
    full_row = summary_df[summary_df["variant"] == "full"].iloc[0]
    gdp_row = summary_df[summary_df["variant"] == "gdp_history_only"].iloc[0]
    macro_row = summary_df[summary_df["variant"] == "macro_only"].iloc[0]

    lines = [
        "# Feature ablation: does removing the macroeconomic predictors cost XGBoost anything?",
        "",
        "Confirmatory diagnostic, not a model change. Refits XGBoost on the frozen 2000-2025 "
        "dataset under expanding-window CV with the same cached hyperparameters as the main "
        "evaluation, on three non-overlapping feature sets.",
        "",
        "## Accuracy comparison (expanding-window CV, mean across 8 folds)",
        "",
        "| Variant | Features | RMSE | MAE |",
        "|---|---|---|---|",
        f"| Full feature set (baseline) | {full_row['n_features']} | "
        f"{full_row['rmse']:.4f} | {full_row['mae']:.4f} |",
        f"| GDP-history only | {gdp_row['n_features']} | "
        f"{gdp_row['rmse']:.4f} | {gdp_row['mae']:.4f} |",
        f"| Macro only | {macro_row['n_features']} | "
        f"{macro_row['rmse']:.4f} | {macro_row['mae']:.4f} |",
        "",
        f"GDP-history features: {gdp_row['features']}",
        "",
        f"Macro features: {macro_row['features']}",
        "",
    ]
    out_path.write_text("\n".join(lines))
    logger.success("wrote {}", out_path)


def main() -> None:
    configure_logging()
    predictions_df, summary_df = run_feature_ablation()

    _PREDICTIONS_OUT.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_parquet(_PREDICTIONS_OUT)
    logger.success("wrote {}", _PREDICTIONS_OUT)

    _write_report(summary_df)

    full_row = summary_df[summary_df["variant"] == "full"].iloc[0]
    logger.info(
        "full-feature RMSE={:.4f}, MAE={:.4f} (recorded main-evaluation RMSE: 2.397391)",
        full_row["rmse"],
        full_row["mae"],
    )


if __name__ == "__main__":
    main()
