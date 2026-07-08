"""XGBoost gradient boosting in the shared ForecastingModel interface.

Conservative defaults (max_depth=3, learning_rate=0.1, n_estimators=200)
suit the 104-row dataset. Trees do not need scaling, so the Pipeline
holds the model alone, with step name "xgboost" for tuning syntax.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.models.interface import ForecastingModel


def build_xgboost_pipeline(
    max_depth: int = 3,
    learning_rate: float = 0.1,
    n_estimators: int = 200,
    random_state: int = 42,
    n_jobs: int = 1,
) -> Pipeline:
    """Returns an unfitted single-step Pipeline holding an XGBRegressor.

    Step name is "xgboost" so tuning can target params as
    xgboost__max_depth. n_jobs defaults to 1 for deterministic runs.
    """
    return Pipeline(
        [
            (
                "xgboost",
                XGBRegressor(
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
            )
        ]
    )


class XGBForecastingModel(ForecastingModel):
    """XGBoost in the shared ForecastingModel interface.

    Thin wrapper around the Pipeline from build_xgboost_pipeline; every
    behaviour except get_params delegates to it.
    """

    def __init__(
        self,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._pipeline: Pipeline = build_xgboost_pipeline(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_state=random_state,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBForecastingModel":
        """Fits the pipeline on the given features and target. Returns self for chaining."""
        self._pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts the target for the given features. Returns one value per row."""
        return self._pipeline.predict(X)

    def get_params(self) -> dict[str, Any]:
        """Returns the four hyperparameters as a dict."""
        return {
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
        }

    def get_estimator(self) -> XGBRegressor:
        """Returns the fitted XGBRegressor, for SHAP and other estimator-level access."""
        return self._pipeline.named_steps["xgboost"]
