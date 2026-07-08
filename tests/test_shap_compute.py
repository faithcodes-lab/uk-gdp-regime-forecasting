"""Tests for src/explainability/shap_compute.py."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from src.explainability.shap_compute import compute_shap_values, load_best_model, load_global_X
from src.models.lightgbm_model import LGBMForecastingModel
from src.models.xgboost_model import XGBForecastingModel


@pytest.fixture
def small_xgb_model() -> tuple[XGBForecastingModel, pd.DataFrame]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(30, 4)), columns=["a", "b", "c", "d"])
    y = X["a"] * 2 + X["b"] - X["c"] + rng.normal(scale=0.1, size=30)
    model = XGBForecastingModel(max_depth=2, n_estimators=20).fit(X, y)
    return model, X


def test_compute_shap_values_shape_matches_input(small_xgb_model):
    model, X = small_xgb_model
    explanation = compute_shap_values(model, X)
    assert explanation.values.shape == X.shape


def test_shap_values_sum_to_prediction(small_xgb_model):
    model, X = small_xgb_model
    explanation = compute_shap_values(model, X)
    reconstructed = explanation.values.sum(axis=1) + explanation.base_values
    predictions = model.predict(X)
    np.testing.assert_allclose(reconstructed, predictions, atol=1e-4)


def test_compute_shap_values_deterministic(small_xgb_model):
    model, X = small_xgb_model
    first = compute_shap_values(model, X).values
    second = compute_shap_values(model, X).values
    np.testing.assert_array_equal(first, second)


def test_load_best_model_picks_lower_rmse_gradient_booster(tmp_path, monkeypatch):
    metrics = pd.DataFrame(
        {
            "model": ["xgboost", "lightgbm", "ridge", "arima"],
            "scheme": ["expanding_window"] * 4,
            "mean_rmse": [2.4, 2.5, 2.8, 4.8],
        }
    )
    metrics_path = tmp_path / "aggregated.csv"
    metrics.to_csv(metrics_path, index=False)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(10, 2)), columns=["a", "b"])
    y = X["a"] + rng.normal(scale=0.1, size=10)
    fitted = XGBForecastingModel(max_depth=2, n_estimators=5).fit(X, y)
    joblib.dump(fitted, models_dir / "xgboost.joblib")

    import src.explainability.shap_compute as shap_compute

    monkeypatch.setattr(shap_compute, "_MODELS_DIR", models_dir)
    model, name = load_best_model(metrics_path=metrics_path)

    assert name == "xgboost"
    assert isinstance(model, XGBForecastingModel)


def test_load_best_model_ignores_non_gradient_boosting_models(tmp_path, monkeypatch):
    metrics = pd.DataFrame(
        {
            "model": ["xgboost", "lightgbm", "ridge", "arima"],
            "scheme": ["expanding_window"] * 4,
            "mean_rmse": [2.4, 999.0, 0.1, 0.1],
        }
    )
    metrics_path = tmp_path / "aggregated.csv"
    metrics.to_csv(metrics_path, index=False)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(10, 2)), columns=["a", "b"])
    y = X["a"] + rng.normal(scale=0.1, size=10)
    fitted = XGBForecastingModel(max_depth=2, n_estimators=5).fit(X, y)
    joblib.dump(fitted, models_dir / "xgboost.joblib")

    import src.explainability.shap_compute as shap_compute

    monkeypatch.setattr(shap_compute, "_MODELS_DIR", models_dir)
    _, name = load_best_model(metrics_path=metrics_path)

    assert name == "xgboost"


def test_lightgbm_get_estimator_is_usable_by_treeshap():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(30, 3)), columns=["a", "b", "c"])
    y = X["a"] - X["b"] + rng.normal(scale=0.1, size=30)
    model = LGBMForecastingModel(max_depth=2, n_estimators=20).fit(X, y)

    explanation = compute_shap_values(model, X)

    assert explanation.values.shape == X.shape


def test_load_best_model_raises_when_no_gradient_boosting_rows(tmp_path):
    metrics = pd.DataFrame(
        {"model": ["ridge", "arima"], "scheme": ["expanding_window"] * 2, "mean_rmse": [2.8, 4.8]}
    )
    metrics_path = tmp_path / "aggregated.csv"
    metrics.to_csv(metrics_path, index=False)

    with pytest.raises(ValueError, match="No gradient boosting rows"):
        load_best_model(metrics_path=metrics_path)


@pytest.mark.integration
def test_load_global_X_matches_the_frozen_training_matrix():
    X = load_global_X()

    assert len(X) == 103
    assert "date" not in X.columns
    assert "regime" not in X.columns
    assert "gdp_growth" in X.columns
