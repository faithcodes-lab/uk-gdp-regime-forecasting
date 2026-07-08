"""Loads the best gradient boosting model and computes global SHAP values.

SHAP attributions describe how the model reasons, not what causes GDP to
move in the real world. Every function here returns exploratory
attributions, not causal estimates. Uses TreeSHAP, which is exact and
deterministic for tree models, unlike sampling-based methods such as
KernelSHAP.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import shap

from src.models.train_all import _load_dataset, _prepare_sklearn_Xy

_AGGREGATED_METRICS = Path("results/metrics/aggregated.csv")
_MODELS_DIR = Path("results/models")
_GRADIENT_BOOSTING_MODELS = ("xgboost", "lightgbm")
_PRIMARY_SCHEME = "expanding_window"


def load_best_model(metrics_path: Path | None = None) -> tuple[object, str]:
    """Returns the best gradient boosting model and its name, ranked by expanding-window mean RMSE."""
    path = metrics_path or _AGGREGATED_METRICS
    df = pd.read_csv(path)
    candidates = df[
        (df["model"].isin(_GRADIENT_BOOSTING_MODELS)) & (df["scheme"] == _PRIMARY_SCHEME)
    ]
    if candidates.empty:
        raise ValueError(f"No gradient boosting rows for scheme {_PRIMARY_SCHEME!r} in {path}")
    best_name = str(candidates.loc[candidates["mean_rmse"].idxmin(), "model"])
    model = joblib.load(_MODELS_DIR / f"{best_name}.joblib")
    return model, best_name


def load_global_X() -> pd.DataFrame:
    """Returns the same feature matrix the reserved full-history models were fit on."""
    df = _load_dataset()
    df = df.dropna().reset_index(drop=True)
    X, _ = _prepare_sklearn_Xy(df)
    return X


def load_global_regimes() -> pd.Series:
    """Returns the regime label for each row of load_global_X(), same length and row order.

    The frozen dataset's last row is dropped when building X, since its
    shifted target is unknown, so this drops that same last row rather than
    using config/regimes.yaml's quarter counts directly.
    """
    df = _load_dataset()
    df = df.dropna().reset_index(drop=True)
    return df["regime"].iloc[:-1].reset_index(drop=True)


def compute_shap_values(model: object, X: pd.DataFrame) -> shap.Explanation:
    """Returns TreeSHAP attributions for X. Exploratory, not causal."""
    explainer = shap.TreeExplainer(model.get_estimator())
    return explainer(X)
