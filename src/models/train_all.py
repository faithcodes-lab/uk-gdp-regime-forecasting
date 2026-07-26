"""Master training script for the four forecasting models.

Single entry point: `python -m src.models.train_all [--retune]`. Loads the
processed parquet dataset, tunes each model (or loads cached best params),
trains the final models on the full dataset, and persists `.joblib` files
and metadata JSON to `results/models/`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm
import pandas as pd
import sklearn
import statsmodels
import xgboost
from loguru import logger

from src.models.arima import ARIMAModel, select_arima_order
from src.models.lightgbm_model import LGBMForecastingModel
from src.models.ridge import RidgeForecastingModel
from src.models.tune import (
    load_tuning_result,
    save_tuning_result,
    tune_lightgbm,
    tune_ridge,
    tune_xgboost,
)
from src.models.xgboost_model import XGBForecastingModel

_DATA_PATH = Path("data/processed/final_dataset.parquet")
_TUNING_DIR = Path("results/tuning")
_MODELS_DIR = Path("results/models")
_TARGET_COLUMN = "gdp_growth"
_DROP_COLUMNS = ["date", "regime"]
_RANDOM_STATE = 42
_TUNING_SPLIT_RATIO = 0.75


def _load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Loads the parquet dataset; raises FileNotFoundError pointing at `make data` if missing."""
    path = path or _DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `make data` first to build the dataset."
        )
    return pd.read_parquet(path)


def _dataset_hash(path: Path) -> str:
    """Returns the md5 hash of the parquet file bytes."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _prepare_sklearn_Xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Builds X and shifted y for the sklearn models.

    X drops date and regime; gdp_growth stays as an autoregressive feature.
    y is gdp_growth shifted by -1 so X[t] predicts y[t+1]. The final row
    (where shifted y is NaN) is dropped from both X and y.
    """
    X = df.drop(columns=_DROP_COLUMNS).reset_index(drop=True)
    y = df[_TARGET_COLUMN].shift(-1).reset_index(drop=True)
    # drop the final row where the shifted target is unknown
    X = X.iloc[:-1].reset_index(drop=True)
    y = y.iloc[:-1].reset_index(drop=True)
    return X, y


def _prepare_arima_y(df: pd.DataFrame) -> pd.Series:
    """Returns the unshifted gdp_growth series for ARIMA."""
    return df[_TARGET_COLUMN].reset_index(drop=True)


def _library_versions() -> dict[str, str]:
    """Returns version strings of the modelling libraries for metadata."""
    return {
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__,
        "statsmodels": statsmodels.__version__,
    }


def _tuning_cache_path(name: str) -> Path:
    return _TUNING_DIR / f"{name}_best_params.json"


def _model_path(name: str) -> Path:
    return _MODELS_DIR / f"{name}.joblib"


def _meta_path(name: str) -> Path:
    return _MODELS_DIR / f"{name}_meta.json"


def _get_or_run_tuning(
    name: str,
    tuner_callable: Callable[[], dict[str, Any]],
    retune: bool,
) -> dict[str, Any]:
    """Returns the tuning result for `name`, from cache when allowed or by running the tuner."""
    cache_path = _tuning_cache_path(name)
    if not retune:
        cached = load_tuning_result(cache_path)
        if cached is not None:
            logger.info(f"[{name}] using cached tuning result from {cache_path}")
            return cached
    logger.info(f"[{name}] tuning (this may take some minutes)")
    result = tuner_callable()
    save_tuning_result(result, cache_path)
    return result


def _build_arima_result(y: pd.Series) -> dict[str, Any]:
    """Runs select_arima_order on the first 75% of y and returns a tuning result dict."""
    n_tune = int(_TUNING_SPLIT_RATIO * len(y))
    order = select_arima_order(y.iloc[:n_tune])
    return {
        "model": "arima",
        "best_params": {"order": list(order), "seasonal_order": [0, 0, 0, 0]},
        "best_score": None,
        "scoring": "aic",
        "n_iter": None,
        "n_splits": None,
        "random_state": _RANDOM_STATE,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def _train_ridge(
    X: pd.DataFrame, y: pd.Series, best_params: dict[str, Any]
) -> RidgeForecastingModel:
    return RidgeForecastingModel(alpha=best_params["ridge__alpha"], random_state=_RANDOM_STATE).fit(
        X, y
    )


def _train_xgboost(
    X: pd.DataFrame, y: pd.Series, best_params: dict[str, Any]
) -> XGBForecastingModel:
    return XGBForecastingModel(
        max_depth=best_params["xgboost__max_depth"],
        learning_rate=best_params["xgboost__learning_rate"],
        n_estimators=best_params["xgboost__n_estimators"],
        random_state=_RANDOM_STATE,
    ).fit(X, y)


def _train_lightgbm(
    X: pd.DataFrame, y: pd.Series, best_params: dict[str, Any]
) -> LGBMForecastingModel:
    return LGBMForecastingModel(
        max_depth=best_params["lightgbm__max_depth"],
        learning_rate=best_params["lightgbm__learning_rate"],
        n_estimators=best_params["lightgbm__n_estimators"],
        min_child_samples=best_params["lightgbm__min_child_samples"],
        random_state=_RANDOM_STATE,
    ).fit(X, y)


def _train_arima(X: pd.DataFrame, y: pd.Series, best_params: dict[str, Any]) -> ARIMAModel:
    # JSON round-trip turns the order tuple into a list; restore tuples for ARIMAModel
    order = tuple(best_params["order"])
    seasonal_order = tuple(best_params["seasonal_order"])
    return ARIMAModel(order=order, seasonal_order=seasonal_order).fit(X, y)


def _persist(
    name: str, model: Any, best_params: dict[str, Any], dataset_hash: str, n_rows: int
) -> None:
    """Saves the model as joblib and writes a metadata JSON next to it."""
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, _model_path(name))
    meta = {
        "model": name,
        "best_params": best_params,
        "library_versions": _library_versions(),
        "training_timestamp": datetime.now(UTC).isoformat(),
        "dataset_hash_md5": dataset_hash,
        "n_training_rows": n_rows,
        "random_state": _RANDOM_STATE,
    }
    _meta_path(name).write_text(json.dumps(meta, indent=2))


def main(retune: bool = False) -> None:
    """End-to-end: load, tune-or-cache, train, persist each of the four models."""
    logger.info("loading dataset")
    df = _load_dataset()
    df = df.dropna().reset_index(drop=True)
    dataset_hash = _dataset_hash(_DATA_PATH)
    logger.info(f"dataset {len(df)} rows, md5 {dataset_hash}")

    X_sklearn, y_sklearn = _prepare_sklearn_Xy(df)
    y_arima = _prepare_arima_y(df)

    _TUNING_DIR.mkdir(parents=True, exist_ok=True)

    ridge_result = _get_or_run_tuning("ridge", lambda: tune_ridge(X_sklearn, y_sklearn), retune)
    ridge_model = _train_ridge(X_sklearn, y_sklearn, ridge_result["best_params"])
    _persist("ridge", ridge_model, ridge_result["best_params"], dataset_hash, len(X_sklearn))

    xgb_result = _get_or_run_tuning("xgboost", lambda: tune_xgboost(X_sklearn, y_sklearn), retune)
    xgb_model = _train_xgboost(X_sklearn, y_sklearn, xgb_result["best_params"])
    _persist("xgboost", xgb_model, xgb_result["best_params"], dataset_hash, len(X_sklearn))

    lgbm_result = _get_or_run_tuning(
        "lightgbm", lambda: tune_lightgbm(X_sklearn, y_sklearn), retune
    )
    lgbm_model = _train_lightgbm(X_sklearn, y_sklearn, lgbm_result["best_params"])
    _persist("lightgbm", lgbm_model, lgbm_result["best_params"], dataset_hash, len(X_sklearn))

    arima_result = _get_or_run_tuning("arima", lambda: _build_arima_result(y_arima), retune)
    arima_model = _train_arima(X_sklearn, y_arima, arima_result["best_params"])
    _persist("arima", arima_model, arima_result["best_params"], dataset_hash, len(y_arima))

    logger.success(f"all four models persisted to {_MODELS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all four forecasting models.")
    parser.add_argument(
        "--retune",
        action="store_true",
        help="Force re-tuning of all models, ignoring any cached best-params.",
    )
    args = parser.parse_args()
    main(retune=args.retune)
