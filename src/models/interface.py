"""Shared interface every Sprint 3 forecasting model implements.

Sprint 3 trains four models (ARIMA, Ridge, XGBoost, LightGBM) under one
CV framework, so all four conform to the same small interface: fit,
predict, get_params. ARIMA conforms by ignoring X.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class ForecastingModel(ABC):
    """Shared interface for every Sprint 3 forecasting model."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ForecastingModel":
        """Trains the model on the given features and target. Returns self for chaining."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts the target for the given features. Returns one value per row."""

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Returns the model's hyperparameters as a dict, for logging and reproducibility records."""
