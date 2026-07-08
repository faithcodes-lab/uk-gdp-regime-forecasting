"""Quantifies SHAP ranking stability across regimes via Spearman rank correlation.

Rankings, not raw SHAP magnitudes, are compared: rank captures which
features matter most, not by how much, so the comparison is not
confounded by GDP volatility differences across regimes. Exploratory, not
causal. Bootstrap CIs on any pair involving a small-sample regime (GFC or
COVID, n=6 each) are the honest quantification of uncertainty those
regimes require, not a flaw in the method: they reveal uncertainty
already present in 6 observations, they do not manufacture certainty.
Stability bands follow Akoglu, H. (2018) 'User's guide to correlation
coefficients', Turkish Journal of Emergency Medicine, 18(3), pp. 91-93.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

_STABLE_THRESHOLD = 0.6
_MODERATE_THRESHOLD = 0.3


def pairwise_spearman_matrix(rankings: pd.DataFrame) -> pd.DataFrame:
    """Returns a regime-by-regime Spearman rho matrix from the feature-by-regime rank matrix."""
    regimes = list(rankings.columns)
    matrix = pd.DataFrame(index=regimes, columns=regimes, dtype=float)
    for regime_a in regimes:
        for regime_b in regimes:
            rho, _ = spearmanr(rankings[regime_a], rankings[regime_b])
            matrix.loc[regime_a, regime_b] = rho
    return matrix


def classify_stability(rho: float) -> str:
    """Classifies a Spearman rho by the Akoglu (2018) bands: stable, moderately stable, or unstable."""
    if rho > _STABLE_THRESHOLD:
        return "stable"
    if rho > _MODERATE_THRESHOLD:
        return "moderately stable"
    return "unstable"


def bootstrap_rankings(
    model: object,
    X_regime: pd.DataFrame,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """Returns an (n_bootstrap, n_features) array of rankings from resampling X_regime with replacement.

    Exploratory: this quantifies the uncertainty a small regime's ranking
    already has, it does not eliminate that uncertainty.
    """
    rng = np.random.default_rng(random_state)
    n = len(X_regime)
    explainer = shap.TreeExplainer(model.get_estimator())
    n_features = X_regime.shape[1]
    out = np.empty((n_bootstrap, n_features))
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        explanation = explainer(X_regime.iloc[idx])
        mean_abs = np.abs(explanation.values).mean(axis=0)
        out[i] = pd.Series(mean_abs).rank(ascending=False, method="average").to_numpy()
    return out


def bootstrap_spearman_ci(
    rankings_a: np.ndarray, rankings_b: np.ndarray, n_bootstrap: int = 1000
) -> tuple[float, float]:
    """Returns the 95% CI for Spearman rho between two regimes' bootstrap ranking distributions.

    rankings_a and rankings_b are each an (n_bootstrap, n_features) array
    from bootstrap_rankings. Row i of one is paired with row i of the
    other to get one rho per bootstrap draw, then the 2.5th and 97.5th
    percentile of that distribution is the CI. Exploratory: for pairs
    involving a small-sample regime this CI is often wide, which honestly
    reflects the uncertainty already present in 6 to 8 observations.
    """
    n = min(len(rankings_a), len(rankings_b), n_bootstrap)
    rhos = np.array([spearmanr(rankings_a[i], rankings_b[i])[0] for i in range(n)])
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))
