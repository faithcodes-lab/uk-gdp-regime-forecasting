"""Ridge regression as the linear multivariate baseline.

A Pipeline of StandardScaler then Ridge, so the scaler is fit on the
training portion of each CV fold automatically. Step names are
"scaler" and "ridge" so tuning can target ridge__alpha.

RidgeForecastingModel is a thin wrapper that exposes the shared
ForecastingModel interface. The caller drops the regime column from X
before fitting and shifts features so row i predicts y at t+1; this
module does not handle either.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.interface import ForecastingModel


def build_ridge_pipeline(alpha: float = 1.0, random_state: int = 42) -> Pipeline:
    """Returns an unfitted Pipeline that scales features and then fits Ridge.

    Caller fits per fold inside a CV loop. The two named steps are
    "scaler" (StandardScaler) and "ridge" (Ridge).
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, random_state=random_state)),
        ]
    )


class RidgeForecastingModel(ForecastingModel):
    """Ridge regression in the shared ForecastingModel interface.

    Thin wrapper around the Pipeline from build_ridge_pipeline; every
    behaviour except get_params delegates to it.
    """

    def __init__(self, alpha: float = 1.0, random_state: int = 42) -> None:
        self.alpha = alpha
        self.random_state = random_state
        self._pipeline: Pipeline = build_ridge_pipeline(alpha=alpha, random_state=random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeForecastingModel":
        """Fits the pipeline on the given features and target. Returns self for chaining."""
        self._pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts the target for the given features. Returns one value per row."""
        return self._pipeline.predict(X)

    def get_params(self) -> dict[str, Any]:
        """Returns alpha and random_state as a dict."""
        return {"alpha": self.alpha, "random_state": self.random_state}
