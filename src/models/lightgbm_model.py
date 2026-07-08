"""LightGBM gradient boosting in the shared ForecastingModel interface.

Conservative defaults (max_depth=3, learning_rate=0.1, n_estimators=200,
min_child_samples=5) suit the 104-row dataset. The min_child_samples=5
override matters: LightGBM's library default is 20, too high for a
quarterly series of this length. Trees do not need scaling, so the
Pipeline holds the model alone, with step name "lightgbm" for tuning
syntax.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline

from src.models.interface import ForecastingModel


def build_lightgbm_pipeline(
    max_depth: int = 3,
    learning_rate: float = 0.1,
    n_estimators: int = 200,
    min_child_samples: int = 5,
    random_state: int = 42,
    n_jobs: int = 1,
) -> Pipeline:
    """Returns an unfitted single-step Pipeline holding an LGBMRegressor.

    Step name is "lightgbm" so tuning can target params as
    lightgbm__max_depth. n_jobs defaults to 1 for deterministic runs.
    min_child_samples defaults to 5 because the library default of 20
    is too restrictive for a 104-row dataset.
    """
    return Pipeline(
        [
            (
                "lightgbm",
                LGBMRegressor(
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    min_child_samples=min_child_samples,
                    random_state=random_state,
                    n_jobs=n_jobs,
                    verbose=-1,
                ),
            )
        ]
    )


class LGBMForecastingModel(ForecastingModel):
    """LightGBM in the shared ForecastingModel interface.

    Thin wrapper around the Pipeline from build_lightgbm_pipeline; every
    behaviour except get_params delegates to it.
    """

    def __init__(
        self,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        n_estimators: int = 200,
        min_child_samples: int = 5,
        random_state: int = 42,
    ) -> None:
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_child_samples = min_child_samples
        self.random_state = random_state
        self._pipeline: Pipeline = build_lightgbm_pipeline(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_samples=min_child_samples,
            random_state=random_state,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMForecastingModel":
        """Fits the pipeline on the given features and target. Returns self for chaining."""
        self._pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts the target for the given features. Returns one value per row."""
        return self._pipeline.predict(X)

    def get_params(self) -> dict[str, Any]:
        """Returns the five hyperparameters as a dict."""
        return {
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "min_child_samples": self.min_child_samples,
            "random_state": self.random_state,
        }

    def get_estimator(self) -> LGBMRegressor:
        """Returns the fitted LGBMRegressor, for SHAP and other estimator-level access."""
        return self._pipeline.named_steps["lightgbm"]
