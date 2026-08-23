"""LASSO comparison: does a feature-selection model find exploitable macro signal?

Confirmatory diagnostic, not a model change. LASSO drives some
coefficients to exactly zero as it fits, so unlike XGBoost it actively
selects features rather than merely being free to ignore them. If the
macroeconomic predictors carried usable forward-looking signal, a model
built to seek and keep the useful ones should both retain them and
forecast well.

Setup mirrors the Ridge baseline for comparability: frozen 2000-2025
dataset, the same one-step-ahead shift, a StandardScaler-plus-estimator
pipeline, alpha tuned by RandomizedSearchCV over the same 20 logspace
values on the first 75% of the data with expanding-window inner
cross-validation, then evaluated under the same 8-fold expanding-window
scheme as the main results with a fresh scaler and LASSO fit inside
each fold. Kept/zeroed coefficients are read from a separate fit on the
full history at the tuned alpha, since per-fold coefficients are not
directly comparable across folds of different sizes.

Run with
    PYTHONPATH=. python scripts/lasso_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Lasso
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import compute_mae, compute_rmse
from src.logging_setup import configure_logging
from src.models.cv import expanding_window_splits
from src.models.train_all import _DATA_PATH, _prepare_sklearn_Xy

_REPORT_OUT = Path("results/lasso-comparison.md")
_TUNING_SPLIT_RATIO = 0.75
_ALPHAS = [float(x) for x in np.logspace(-3, 2, 20)]
_RANDOM_STATE = 42


def _build_lasso_pipeline(alpha: float = 1.0, random_state: int = 42) -> Pipeline:
    """Returns an unfitted Pipeline that scales features and then fits LASSO."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lasso", Lasso(alpha=alpha, random_state=random_state, max_iter=10_000)),
        ]
    )


def tune_lasso_alpha(X: pd.DataFrame, y: pd.Series, random_state: int = _RANDOM_STATE) -> float:
    """Returns the best LASSO alpha via RandomizedSearchCV on the first 75% of X, y.

    Mirrors src.models.tune.tune_ridge exactly (same alphas, same 75%
    tuning window, same expanding-window inner CV, same neg-RMSE
    scoring) so the two penalty types are compared on equal footing.
    """
    n_tune = int(_TUNING_SPLIT_RATIO * len(X))
    X_tune, y_tune = X.iloc[:n_tune], y.iloc[:n_tune]
    splits = list(expanding_window_splits(X_tune, n_splits=5, test_size=4))
    search = RandomizedSearchCV(
        estimator=_build_lasso_pipeline(random_state=random_state),
        param_distributions={"lasso__alpha": _ALPHAS},
        n_iter=20,
        cv=splits,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
        n_jobs=1,
    )
    search.fit(X_tune, y_tune)
    return float(search.best_params_["lasso__alpha"])


def _cross_validate(
    X: pd.DataFrame, y: pd.Series, alpha: float, splits: list[tuple[np.ndarray, np.ndarray]]
) -> pd.DataFrame:
    """Runs expanding-window CV for LASSO at a fixed alpha, refitting per fold."""
    rows: list[dict] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        pipeline = _build_lasso_pipeline(alpha=alpha, random_state=_RANDOM_STATE)
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        for pos, idx in enumerate(test_idx):
            rows.append(
                {
                    "fold_idx": fold_idx,
                    "y_true": float(y_test.iloc[pos]),
                    "y_pred": float(y_pred[pos]),
                }
            )
    return pd.DataFrame(rows)


def _full_data_coefficients(X: pd.DataFrame, y: pd.Series, alpha: float) -> pd.Series:
    """Returns LASSO coefficients from a single fit on the full history, sorted by magnitude."""
    pipeline = _build_lasso_pipeline(alpha=alpha, random_state=_RANDOM_STATE)
    pipeline.fit(X, y)
    coefs = pipeline.named_steps["lasso"].coef_
    return pd.Series(coefs, index=X.columns).sort_values(key=np.abs, ascending=False)


def run_lasso_comparison(
    data_path: Path | str = _DATA_PATH,
) -> tuple[float, float, float, pd.Series, pd.DataFrame]:
    """Returns (alpha, rmse, mae, full-data coefficients, per-observation predictions)."""
    df = pd.read_parquet(data_path)
    X, y = _prepare_sklearn_Xy(df)

    alpha = tune_lasso_alpha(X, y)
    splits = expanding_window_splits(X, n_splits=8, test_size=4)
    preds = _cross_validate(X, y, alpha, splits)

    per_fold_rmse = preds.groupby("fold_idx").apply(
        lambda g: compute_rmse(g["y_true"].to_numpy(), g["y_pred"].to_numpy())
    )
    per_fold_mae = preds.groupby("fold_idx").apply(
        lambda g: compute_mae(g["y_true"].to_numpy(), g["y_pred"].to_numpy())
    )
    rmse = float(per_fold_rmse.mean())
    mae = float(per_fold_mae.mean())

    coefficients = _full_data_coefficients(X, y, alpha)
    return alpha, rmse, mae, coefficients, preds


def _write_report(
    alpha: float, rmse: float, mae: float, coefficients: pd.Series, out_path: Path = _REPORT_OUT
) -> None:
    kept = coefficients[coefficients != 0]
    zeroed = coefficients[coefficients == 0]
    macro_kept = [f for f in kept.index if not f.startswith("gdp_")]
    gdp_kept = [f for f in kept.index if f.startswith("gdp_")]

    lines = [
        "# LASSO comparison: does a feature-selection model find exploitable macro signal?",
        "",
        "Confirmatory diagnostic, not a model change. LASSO is linear regression with an L1 "
        "penalty, which drives some coefficients to exactly zero, so it selects features as it "
        "fits. If the macroeconomic predictors carried usable forward-looking signal, a model "
        "built to seek and keep the useful ones should both retain them and forecast well.",
        "",
        "Setup mirrors the Ridge baseline for comparability: frozen 2000-2025 dataset, the same "
        "one-step-ahead shift, a StandardScaler-plus-estimator pipeline, alpha tuned by "
        "RandomizedSearchCV over 20 logspace values on the first 75% of the data with "
        "expanding-window inner cross-validation and neg-RMSE scoring, then evaluated under the "
        f"same 8-fold expanding-window scheme as the main results with a fresh scaler and LASSO "
        f"fit inside each fold on that fold's training rows only. Tuned alpha was {alpha:.4f}.",
        "",
        "## Accuracy (expanding-window CV)",
        "",
        "| Model | RMSE | MAE |",
        "|---|---|---|",
        f"| LASSO | {rmse:.3f} | {mae:.3f} |",
        "",
        "For the full multi-model comparison (XGBoost, Ridge, LightGBM, ARIMA, and the feature "
        "ablation variants), see results/feature-ablation.md and results/metrics/aggregated.csv.",
        "",
        "## Which features LASSO kept (full-data fit, tuned alpha)",
        "",
        f"LASSO kept {len(kept)} of {len(coefficients)} features with nonzero coefficients "
        f"({len(gdp_kept)} GDP-history, {len(macro_kept)} macro):",
        "",
        f"- Kept ({len(kept)}): " + ", ".join(f"{f} ({v:.2f})" for f, v in kept.items()) + ".",
        f"- Zeroed ({len(zeroed)}): " + ", ".join(zeroed.index) + ".",
        "",
        "## Reading",
        "",
        "LASSO assigns substantial in-sample weight to macro features — its single largest "
        f"coefficient is a macro one ({kept.index[0]}, {kept.iloc[0]:.2f}) — so a linear "
        "feature-selector left to fit thinks the macro features are useful. But those macro "
        "coefficients buy no out-of-sample accuracy: LASSO is the worst forecaster of every model "
        "tested except ARIMA. The macro signal it latches onto is in-sample fit that does not "
        "generalise to held-out quarters. XGBoost ignores the macro features and forecasts well "
        "on GDP history alone; LASSO uses the macro features and is punished for it out-of-sample. "
        "Both point to the same conclusion from opposite directions: the macro features carry "
        "in-sample fit but no exploitable out-of-sample signal at this sample size (103 training "
        "rows).",
        "",
        "Honest caveat: the GDP-history features are highly correlated with each other, and LASSO "
        "is known to be unstable in which member of a correlated group it retains — it tends to "
        "pick one somewhat arbitrarily and can zero a near-duplicate (here it zeroed "
        "gdp_rolling_mean_4q while keeping gdp_yoy). So the exact selected and zeroed lists above "
        "should not be over-interpreted feature by feature. The robust, reportable result is the "
        "poor out-of-sample RMSE, not the precise membership of the kept set.",
        "",
    ]
    out_path.write_text("\n".join(lines))
    logger.success("wrote {}", out_path)


def main() -> None:
    configure_logging()
    alpha, rmse, mae, coefficients, preds = run_lasso_comparison()

    kept = coefficients[coefficients != 0]
    logger.info("tuned alpha={:.4f} (recorded: 0.0695)", alpha)
    logger.info("LASSO expanding-window RMSE={:.3f} MAE={:.3f} (recorded RMSE=3.947 MAE=3.329)", rmse, mae)
    logger.info(
        "kept {} of {} features, largest |coef|: {} ({:.3f})",
        len(kept),
        len(coefficients),
        kept.index[0],
        kept.iloc[0],
    )

    preds_out = Path("results/predictions/lasso_predictions.parquet")
    preds_out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(preds_out)
    logger.success("wrote {}", preds_out)

    _write_report(alpha, rmse, mae, coefficients)


if __name__ == "__main__":
    main()
