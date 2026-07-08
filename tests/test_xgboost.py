"""Tests for src/models/xgboost_model.py.

The scaler-free pipeline test and the integration test through the
expanding-window splitter are the two that matter most: they prove
trees do not get scaling wired in by accident and that the model
runs cleanly across every CV fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.models.cv import expanding_window_splits
from src.models.interface import ForecastingModel
from src.models.xgboost_model import XGBForecastingModel, build_xgboost_pipeline


def _synthetic_Xy(n: int = 104, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Builds synthetic X (three features) and y for a deterministic linear relationship plus small noise."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
        }
    )
    y = pd.Series(0.5 + 2 * X["a"] - X["b"] + 0.3 * X["c"] + rng.normal(scale=0.1, size=n))
    return X, y


def test_xgb_subclasses_forecasting_model():
    """XGBForecastingModel is a subclass of ForecastingModel."""
    assert issubclass(XGBForecastingModel, ForecastingModel)


def test_xgb_get_params_returns_all_hyperparameters():
    """get_params returns exactly the four hyperparameters with their current values."""
    model = XGBForecastingModel(max_depth=4, learning_rate=0.05, n_estimators=100, random_state=7)
    assert model.get_params() == {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "random_state": 7,
    }


def test_xgb_fit_returns_self():
    """fit returns the model itself so calls can be chained."""
    X, y = _synthetic_Xy(n=30)
    model = XGBForecastingModel()
    returned = model.fit(X, y)
    assert returned is model


def test_xgb_fit_predict_returns_correct_shape():
    """predict returns one value per input row."""
    X, y = _synthetic_Xy(n=50)
    model = XGBForecastingModel().fit(X.iloc[:40], y.iloc[:40])
    preds = model.predict(X.iloc[40:])
    assert preds.shape == (10,)


def test_xgb_pipeline_has_no_scaler_step():
    """The Pipeline holds only the xgboost step, never a StandardScaler.

    Trees are scale-invariant; any scaling wired in here would be a bug
    under the rule that scaling is only for Ridge.
    """
    pipe = build_xgboost_pipeline()
    assert list(pipe.named_steps.keys()) == ["xgboost"]
    for step in pipe.named_steps.values():
        assert not isinstance(step, StandardScaler)


def test_xgb_conservative_defaults_applied():
    """Default hyperparameters fall inside the conservative ranges agreed for the 104-row dataset."""
    pipe = build_xgboost_pipeline()
    xgb_step = pipe.named_steps["xgboost"]
    assert isinstance(xgb_step, XGBRegressor)
    assert 2 <= xgb_step.max_depth <= 4
    assert 0.01 <= xgb_step.learning_rate <= 0.1
    assert 50 <= xgb_step.n_estimators <= 500


def test_xgb_deterministic_with_fixed_seed():
    """Two XGBForecastingModel instances with the same seed and data produce identical predictions."""
    X, y = _synthetic_Xy(n=60)
    X_train, X_test, y_train = X.iloc[:50], X.iloc[50:], y.iloc[:50]

    a = XGBForecastingModel(random_state=42).fit(X_train, y_train)
    b = XGBForecastingModel(random_state=42).fit(X_train, y_train)
    np.testing.assert_array_equal(a.predict(X_test), b.predict(X_test))


def test_xgb_works_with_expanding_window_splits():
    """XGBoost runs cleanly across every fold of the expanding-window CV splitter."""
    X, y = _synthetic_Xy(n=104)
    for train_idx, test_idx in expanding_window_splits(X, n_splits=8, test_size=4):
        # fresh pipeline per fold, exactly as in a real CV loop
        model = XGBForecastingModel().fit(X.iloc[train_idx], y.iloc[train_idx])
        assert model.predict(X.iloc[test_idx]).shape == (len(test_idx),)


def test_xgb_predict_before_fit_raises():
    """Calling predict before fit raises sklearn's NotFittedError."""
    X, _ = _synthetic_Xy(n=10)
    with pytest.raises(NotFittedError):
        XGBForecastingModel().predict(X)
