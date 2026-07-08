"""Computes SHAP values within each regime and ranks features by regime.

Per-regime SHAP is the novel contribution: the same TreeSHAP explanation
is partitioned by regime to ask whether the model attributes importance
to the same features across UK economic regimes. Exploratory, not
causal: this describes the model's reasoning, not real-world drivers of
GDP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

_SMALL_SAMPLE_THRESHOLD = 10


def compute_per_regime_shap(
    model: object, X: pd.DataFrame, regimes: pd.Series
) -> dict[str, shap.Explanation]:
    """Returns a dict mapping each regime label to its own SHAP Explanation, computed only on that regime's rows."""
    if len(regimes) != len(X):
        raise ValueError(f"regimes has {len(regimes)} entries but X has {len(X)} rows")
    regimes = pd.Series(regimes).reset_index(drop=True)
    X = X.reset_index(drop=True)
    explainer = shap.TreeExplainer(model.get_estimator())
    return {
        str(regime): explainer(X.loc[(regimes == regime).to_numpy()]) for regime in regimes.unique()
    }


def compute_per_regime_rankings(
    per_regime_shap: dict[str, shap.Explanation],
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Returns a feature-by-regime rank matrix (1 = most important) and small-sample metadata per regime."""
    mean_abs_shap: dict[str, pd.Series] = {}
    metadata: dict[str, dict[str, object]] = {}
    for regime, explanation in per_regime_shap.items():
        mean_abs = np.abs(explanation.values).mean(axis=0)
        mean_abs_shap[regime] = pd.Series(mean_abs, index=explanation.feature_names)
        n_obs = explanation.values.shape[0]
        metadata[regime] = {
            "n_observations": n_obs,
            "small_sample": n_obs < _SMALL_SAMPLE_THRESHOLD,
        }

    importance = pd.DataFrame(mean_abs_shap)
    rankings = importance.rank(ascending=False, method="average")
    return rankings, metadata
