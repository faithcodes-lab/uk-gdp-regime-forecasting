"""Hyperparameter tuning for Ridge, XGBoost, and LightGBM.

Uses RandomizedSearchCV with the expanding-window CV scheme on the first
75% of data. The remaining 25% stays untouched for evaluation. Best
params are cached as JSON so downstream work runs without re-tuning.
ARIMA is not tuned here; its order is selected once via select_arima_order.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from src.models.cv import expanding_window_splits
from src.models.lightgbm_model import build_lightgbm_pipeline
from src.models.ridge import build_ridge_pipeline
from src.models.xgboost_model import build_xgboost_pipeline

_TUNING_SPLIT_RATIO = 0.75
_SCORING = "neg_root_mean_squared_error"


def _tune_window(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Returns the first 75% of X and y for tuning; the remaining 25% is reserved for evaluation."""
    n_tune = int(_TUNING_SPLIT_RATIO * len(X))
    return X.iloc[:n_tune], y.iloc[:n_tune]


def _run_search(
    pipeline,
    param_distributions: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    n_iter: int,
    random_state: int,
) -> RandomizedSearchCV:
    """Runs RandomizedSearchCV with expanding-window CV on the tuning window only."""
    X_tune, y_tune = _tune_window(X, y)
    # materialise to a list because RandomizedSearchCV consumes cv more than once
    splits = list(expanding_window_splits(X_tune, n_splits=n_splits, test_size=4))
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=splits,
        scoring=_SCORING,
        random_state=random_state,
        n_jobs=1,
    )
    search.fit(X_tune, y_tune)
    return search


def _to_jsonable(value: Any) -> Any:
    """Converts numpy scalar types to Python types so the result dict is JSON serialisable."""
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _result_dict(
    model: str,
    best_params: dict[str, Any],
    best_score: float,
    n_iter: int,
    n_splits: int,
    random_state: int,
) -> dict[str, Any]:
    """Builds the cached result dict in the documented schema."""
    return {
        "model": model,
        "best_params": {k: _to_jsonable(v) for k, v in best_params.items()},
        "best_score": float(best_score),
        "scoring": _SCORING,
        "n_iter": n_iter,
        "n_splits": n_splits,
        "random_state": random_state,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def tune_ridge(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_iter: int = 20,
    random_state: int = 42,
    cache_path: Path | str | None = None,
) -> dict[str, Any]:
    """Tunes Ridge alpha over 20 logspace values and returns the best-params result dict."""
    # cast logspace values to Python floats so JSON serialisation stays simple downstream
    alphas = [float(x) for x in np.logspace(-3, 2, 20)]
    pipeline = build_ridge_pipeline(random_state=random_state)
    search = _run_search(
        pipeline=pipeline,
        param_distributions={"ridge__alpha": alphas},
        X=X,
        y=y,
        n_splits=n_splits,
        n_iter=n_iter,
        random_state=random_state,
    )
    result = _result_dict(
        "ridge", search.best_params_, search.best_score_, n_iter, n_splits, random_state
    )
    if cache_path is not None:
        save_tuning_result(result, cache_path)
    return result


def tune_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_iter: int = 30,
    random_state: int = 42,
    cache_path: Path | str | None = None,
) -> dict[str, Any]:
    """Tunes XGBoost over the conservative grid and returns the best-params result dict."""
    param_distributions = {
        "xgboost__max_depth": [2, 3, 4],
        "xgboost__learning_rate": [0.01, 0.05, 0.1],
        "xgboost__n_estimators": [50, 100, 200, 500],
    }
    pipeline = build_xgboost_pipeline(random_state=random_state)
    search = _run_search(
        pipeline=pipeline,
        param_distributions=param_distributions,
        X=X,
        y=y,
        n_splits=n_splits,
        n_iter=n_iter,
        random_state=random_state,
    )
    result = _result_dict(
        "xgboost", search.best_params_, search.best_score_, n_iter, n_splits, random_state
    )
    if cache_path is not None:
        save_tuning_result(result, cache_path)
    return result


def tune_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_iter: int = 30,
    random_state: int = 42,
    cache_path: Path | str | None = None,
) -> dict[str, Any]:
    """Tunes LightGBM over the conservative grid and returns the best-params result dict."""
    param_distributions = {
        "lightgbm__max_depth": [2, 3, 4],
        "lightgbm__learning_rate": [0.01, 0.05, 0.1],
        "lightgbm__n_estimators": [50, 100, 200, 500],
        "lightgbm__min_child_samples": [5, 10],
    }
    pipeline = build_lightgbm_pipeline(random_state=random_state)
    search = _run_search(
        pipeline=pipeline,
        param_distributions=param_distributions,
        X=X,
        y=y,
        n_splits=n_splits,
        n_iter=n_iter,
        random_state=random_state,
    )
    result = _result_dict(
        "lightgbm", search.best_params_, search.best_score_, n_iter, n_splits, random_state
    )
    if cache_path is not None:
        save_tuning_result(result, cache_path)
    return result


def save_tuning_result(result: dict[str, Any], cache_path: Path | str) -> Path:
    """Writes the tuning result as JSON, creating parent directories as needed."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return path


def load_tuning_result(cache_path: Path | str) -> dict[str, Any] | None:
    """Reads a cached tuning result, or returns None if the file does not exist."""
    path = Path(cache_path)
    if not path.exists():
        return None
    return json.loads(path.read_text())
