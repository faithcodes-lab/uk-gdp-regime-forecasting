"""Tests for src/models/interface.py and src/models/ridge.py.

The scaling discipline test is the leakage heart: it proves a
Pipeline-wrapped Ridge fits its StandardScaler on the training portion
only, never the full series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.models.cv import expanding_window_splits
from src.models.interface import ForecastingModel
from src.models.ridge import RidgeForecastingModel, build_ridge_pipeline


def _synthetic_Xy(n: int = 104, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Builds synthetic X (three features at very different scales) and y for a deterministic linear relationship plus small noise."""
    rng = np.random.default_rng(seed)
    # three features at very different scales so StandardScaler has something obvious to do
    X = pd.DataFrame(
        {
            "small": rng.normal(scale=0.01, size=n),
            "medium": rng.normal(scale=1.0, size=n),
            "large": rng.normal(scale=1000.0, size=n),
        }
    )
    y = pd.Series(0.5 + 2 * X["medium"] + 0.001 * X["large"] + rng.normal(scale=0.1, size=n))
    return X, y


def test_ridge_subclasses_forecasting_model():
    """RidgeForecastingModel is a subclass of ForecastingModel."""
    assert issubclass(RidgeForecastingModel, ForecastingModel)


def test_ridge_has_required_abstract_methods():
    """RidgeForecastingModel implements all three abstract methods so it can be instantiated."""
    model = RidgeForecastingModel()
    assert callable(model.fit)
    assert callable(model.predict)
    assert callable(model.get_params)


def test_ridge_get_params_returns_alpha_and_random_state():
    """get_params returns exactly the two hyperparameters with their current values."""
    model = RidgeForecastingModel(alpha=2.5, random_state=7)
    assert model.get_params() == {"alpha": 2.5, "random_state": 7}


def test_ridge_fit_returns_self():
    """fit returns the model itself so calls can be chained."""
    X, y = _synthetic_Xy(n=30)
    model = RidgeForecastingModel()
    returned = model.fit(X, y)
    assert returned is model


def test_ridge_fit_predict_returns_correct_shape():
    """predict returns one value per input row."""
    X, y = _synthetic_Xy(n=50)
    model = RidgeForecastingModel().fit(X.iloc[:40], y.iloc[:40])
    preds = model.predict(X.iloc[40:])
    assert preds.shape == (10,)


def test_ridge_pipeline_steps_are_scaler_then_ridge():
    """The Pipeline has the scaler step before the ridge step, in that order."""
    pipe = build_ridge_pipeline()
    assert list(pipe.named_steps.keys()) == ["scaler", "ridge"]


def test_ridge_pipeline_uses_standard_scaler_and_ridge():
    """The 'scaler' step is a StandardScaler and the 'ridge' step is a Ridge with the given alpha."""
    pipe = build_ridge_pipeline(alpha=2.0)
    assert isinstance(pipe.named_steps["scaler"], StandardScaler)
    assert isinstance(pipe.named_steps["ridge"], Ridge)
    assert pipe.named_steps["ridge"].alpha == 2.0


def test_ridge_pipeline_scales_training_fold_only_not_full_data():
    """Fitting the pipeline on a slice of X scales using only that slice's mean, not the full-X mean.

    Proves the Pipeline avoids leakage when used inside a CV fold:
    scaler.mean_ reflects the training rows, not the test rows.
    """
    X, y = _synthetic_Xy(n=100)
    train_slice = slice(0, 50)
    pipe = build_ridge_pipeline()
    pipe.fit(X.iloc[train_slice], y.iloc[train_slice])

    scaler = pipe.named_steps["scaler"]
    train_mean = X.iloc[train_slice].mean(axis=0).to_numpy()
    full_mean = X.mean(axis=0).to_numpy()

    np.testing.assert_array_almost_equal(scaler.mean_, train_mean)
    # safeguard: full-X mean must differ from training mean, otherwise
    # the previous assertion would pass even if the scaler leaked
    assert not np.allclose(
        train_mean, full_mean
    ), "synthetic data did not produce a train vs full mean gap; pick a different slice"


def test_ridge_deterministic_with_fixed_seed():
    """Two RidgeForecastingModel instances with the same seed and data produce identical predictions."""
    X, y = _synthetic_Xy(n=60)
    X_train, X_test, y_train = X.iloc[:50], X.iloc[50:], y.iloc[:50]

    a = RidgeForecastingModel(random_state=42).fit(X_train, y_train)
    b = RidgeForecastingModel(random_state=42).fit(X_train, y_train)
    np.testing.assert_array_equal(a.predict(X_test), b.predict(X_test))


def test_ridge_works_with_expanding_window_splits():
    """Ridge runs cleanly across every fold of the expanding-window CV splitter."""
    X, y = _synthetic_Xy(n=104)
    for train_idx, test_idx in expanding_window_splits(X, n_splits=8, test_size=4):
        # fresh pipeline per fold, exactly as in a real CV loop
        model = RidgeForecastingModel().fit(X.iloc[train_idx], y.iloc[train_idx])
        assert model.predict(X.iloc[test_idx]).shape == (len(test_idx),)


def test_ridge_predict_before_fit_raises():
    """Calling predict before fit raises sklearn's NotFittedError."""
    X, _ = _synthetic_Xy(n=10)
    with pytest.raises(NotFittedError):
        RidgeForecastingModel().predict(X)
