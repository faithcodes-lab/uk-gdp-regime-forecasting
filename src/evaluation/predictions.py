"""Generates per-fold one-step-ahead predictions from the four trained models.

For each of Ridge, XGBoost, LightGBM, and ARIMA, runs the model through
both CV schemes (expanding-window and regime-aligned) and emits a long-
format DataFrame tagged with model, target quarter, regime, fold index,
and scheme. This DataFrame is the input to every downstream evaluation
checkpoint (DM test, per-regime breakdown, aggregation, tables).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.cv import (
    cross_validate_arima,
    expanding_window_splits,
    regime_aligned_splits,
)
from src.models.lightgbm_model import LGBMForecastingModel
from src.models.ridge import RidgeForecastingModel
from src.models.train_all import _prepare_arima_y, _prepare_sklearn_Xy
from src.models.tune import load_tuning_result
from src.models.xgboost_model import XGBForecastingModel

_TUNING_DIR = Path("results/tuning")


def generate_predictions(
    df: pd.DataFrame,
    output_path: Path | str | None = None,
    n_splits: int = 8,
    test_size: int = 4,
) -> pd.DataFrame:
    """Generates per-fold predictions for the four models across both CV schemes.

    Returns a long-format DataFrame with columns model, quarter, regime,
    y_true, y_pred, fold_idx, scheme. If output_path is given, also writes
    the DataFrame as a parquet file.
    """
    # common 103-row target space so all four models predict the same quarters
    target_df = df.iloc[1:].reset_index(drop=True)
    X_sklearn, y_sklearn = _prepare_sklearn_Xy(df)
    y_arima = _prepare_arima_y(df)

    expanding = expanding_window_splits(target_df, n_splits=n_splits, test_size=test_size)
    regime_aligned = regime_aligned_splits(target_df, regime_column="regime")

    ridge_params = load_tuning_result(_TUNING_DIR / "ridge_best_params.json")["best_params"]
    xgb_params = load_tuning_result(_TUNING_DIR / "xgboost_best_params.json")["best_params"]
    lgbm_params = load_tuning_result(_TUNING_DIR / "lightgbm_best_params.json")["best_params"]
    arima_params = load_tuning_result(_TUNING_DIR / "arima_best_params.json")["best_params"]
    arima_order = tuple(arima_params["order"])

    frames: list[pd.DataFrame] = []
    for scheme_name, splits in [
        ("expanding_window", expanding),
        ("regime_aligned", regime_aligned),
    ]:
        frames.append(
            _predict_sklearn_per_scheme(
                "ridge", ridge_params, X_sklearn, y_sklearn, splits, target_df, scheme_name
            )
        )
        frames.append(
            _predict_sklearn_per_scheme(
                "xgboost", xgb_params, X_sklearn, y_sklearn, splits, target_df, scheme_name
            )
        )
        frames.append(
            _predict_sklearn_per_scheme(
                "lightgbm", lgbm_params, X_sklearn, y_sklearn, splits, target_df, scheme_name
            )
        )
        frames.append(
            _predict_arima_per_scheme(arima_order, y_arima, splits, target_df, scheme_name)
        )

    result = pd.concat(frames, ignore_index=True)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(out)

    return result


def _predict_sklearn_per_scheme(
    model_name: str,
    best_params: dict[str, Any],
    X_sklearn: pd.DataFrame,
    y_sklearn: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    target_df: pd.DataFrame,
    scheme_name: str,
) -> pd.DataFrame:
    """Builds the predictions DataFrame for one sklearn model on one CV scheme.

    For each fold, refits the model with the cached best-params on the
    training portion and predicts the test portion. Tags every prediction
    with quarter, regime, y_true from target_df.
    """
    factory_class = {
        "ridge": RidgeForecastingModel,
        "xgboost": XGBForecastingModel,
        "lightgbm": LGBMForecastingModel,
    }[model_name]
    model_kwargs = _sklearn_kwargs_from_best_params(model_name, best_params)

    rows: list[dict[str, Any]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        model = factory_class(**model_kwargs)
        model.fit(X_sklearn.iloc[train_idx], y_sklearn.iloc[train_idx])
        preds = model.predict(X_sklearn.iloc[test_idx])
        for k, target_pos in enumerate(test_idx):
            t = int(target_pos)
            rows.append(
                {
                    "model": model_name,
                    "quarter": target_df["date"].iloc[t],
                    "regime": target_df["regime"].iloc[t],
                    "y_true": float(target_df["gdp_growth"].iloc[t]),
                    "y_pred": float(preds[k]),
                    "fold_idx": fold_idx,
                    "scheme": scheme_name,
                }
            )
    return pd.DataFrame(rows)


def _predict_arima_per_scheme(
    arima_order: tuple[int, int, int],
    y_arima: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    target_df: pd.DataFrame,
    scheme_name: str,
) -> pd.DataFrame:
    """Builds the predictions DataFrame for ARIMA on one CV scheme.

    The +1 offset maps target_df indices (103 rows) to y_arima indices
    (104 rows): predicting target_df.iloc[t] means refitting ARIMA on
    y_arima.iloc[:t+1] and forecasting y_arima.iloc[t+1].
    """
    # shift split indices by +1 so they index into y_arima (104 rows) rather
    # than target_df (103 rows); this is what makes ARIMA predict the same
    # target quarters as the sklearn models
    arima_splits = [(train_idx + 1, test_idx + 1) for train_idx, test_idx in splits]
    arima_preds = cross_validate_arima(y_arima, arima_splits, order=arima_order)

    rows: list[dict[str, Any]] = []
    pred_cursor = 0
    for fold_idx, (_train_idx, test_idx) in enumerate(splits, start=1):
        for target_pos in test_idx:
            t = int(target_pos)
            rows.append(
                {
                    "model": "arima",
                    "quarter": target_df["date"].iloc[t],
                    "regime": target_df["regime"].iloc[t],
                    "y_true": float(target_df["gdp_growth"].iloc[t]),
                    "y_pred": float(arima_preds[pred_cursor]),
                    "fold_idx": fold_idx,
                    "scheme": scheme_name,
                }
            )
            pred_cursor += 1
    return pd.DataFrame(rows)


def _sklearn_kwargs_from_best_params(
    model_name: str, best_params: dict[str, Any]
) -> dict[str, Any]:
    """Strips the sklearn Pipeline step prefix (e.g. ridge__alpha) and returns plain kwargs."""
    if model_name == "ridge":
        return {"alpha": best_params["ridge__alpha"]}
    if model_name == "xgboost":
        return {
            "max_depth": best_params["xgboost__max_depth"],
            "learning_rate": best_params["xgboost__learning_rate"],
            "n_estimators": best_params["xgboost__n_estimators"],
        }
    if model_name == "lightgbm":
        return {
            "max_depth": best_params["lightgbm__max_depth"],
            "learning_rate": best_params["lightgbm__learning_rate"],
            "n_estimators": best_params["lightgbm__n_estimators"],
            "min_child_samples": best_params["lightgbm__min_child_samples"],
        }
    raise ValueError(f"unknown sklearn model: {model_name}")
