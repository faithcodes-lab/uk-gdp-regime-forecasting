"""Tests for src/explainability/regime_shap.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.explainability.regime_shap import compute_per_regime_rankings, compute_per_regime_shap
from src.explainability.shap_compute import (
    compute_shap_values,
    load_best_model,
    load_global_regimes,
    load_global_X,
)
from src.models.xgboost_model import XGBForecastingModel


@pytest.fixture
def small_model_and_data() -> tuple[XGBForecastingModel, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    n = 40
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = X["a"] * 2 - X["b"] + rng.normal(scale=0.1, size=n)
    model = XGBForecastingModel(max_depth=2, n_estimators=20).fit(X, y)
    # Three regimes of uneven size, one deliberately below the small-sample threshold.
    regimes = pd.Series(["stable"] * 20 + ["shock"] * 5 + ["recovery"] * 15)
    return model, X, regimes


def test_compute_per_regime_shap_covers_every_regime(small_model_and_data):
    model, X, regimes = small_model_and_data
    per_regime = compute_per_regime_shap(model, X, regimes)
    assert set(per_regime.keys()) == {"stable", "shock", "recovery"}


def test_compute_per_regime_shap_partitions_row_counts_correctly(small_model_and_data):
    model, X, regimes = small_model_and_data
    per_regime = compute_per_regime_shap(model, X, regimes)
    assert per_regime["stable"].values.shape[0] == 20
    assert per_regime["shock"].values.shape[0] == 5
    assert per_regime["recovery"].values.shape[0] == 15


def test_compute_per_regime_shap_raises_on_length_mismatch(small_model_and_data):
    model, X, regimes = small_model_and_data
    with pytest.raises(ValueError, match="regimes has"):
        compute_per_regime_shap(model, X, regimes.iloc[:-1])


def test_rankings_give_rank_one_to_highest_mean_abs_shap(small_model_and_data):
    model, X, regimes = small_model_and_data
    per_regime = compute_per_regime_shap(model, X, regimes)
    rankings, _ = compute_per_regime_rankings(per_regime)

    assert set(rankings.columns) == {"stable", "shock", "recovery"}
    assert set(rankings.index) == {"a", "b", "c"}
    # Feature "a" has the largest coefficient in the synthetic target, so it
    # should rank first (lowest rank number) in every regime.
    assert (rankings.loc["a"] == 1.0).all()
    # Every column is a valid ranking over 3 features: values 1..3.
    for col in rankings.columns:
        assert sorted(rankings[col].tolist()) == [1.0, 2.0, 3.0]


def test_small_sample_flagging(small_model_and_data):
    model, X, regimes = small_model_and_data
    per_regime = compute_per_regime_shap(model, X, regimes)
    _, metadata = compute_per_regime_rankings(per_regime)

    assert metadata["shock"]["n_observations"] == 5
    assert metadata["shock"]["small_sample"] is True
    assert metadata["stable"]["n_observations"] == 20
    assert metadata["stable"]["small_sample"] is False
    assert metadata["recovery"]["small_sample"] is False


@pytest.mark.integration
def test_per_regime_shap_on_real_model_and_real_regimes():
    """Integration check against the real xgboost.joblib and the real regime column."""
    model, _ = load_best_model()
    X = load_global_X()
    regimes = load_global_regimes()

    per_regime = compute_per_regime_shap(model, X, regimes)

    assert set(per_regime.keys()) == {
        "Pre-GFC Stability",
        "Global Financial Crisis",
        "Post-GFC Recovery",
        "Brexit Transition",
        "COVID-19 Shock",
        "Post-COVID Recovery",
    }
    # Post-COVID Recovery loses its last quarter (2025 Q4) when X drops the
    # final row, so it is 17 here, not the 18 quarters in config/regimes.yaml.
    assert per_regime["Post-COVID Recovery"].values.shape[0] == 17
    assert per_regime["Global Financial Crisis"].values.shape[0] == 6
    assert per_regime["COVID-19 Shock"].values.shape[0] == 6

    rankings, metadata = compute_per_regime_rankings(per_regime)
    assert metadata["Global Financial Crisis"]["small_sample"] is True
    assert metadata["COVID-19 Shock"]["small_sample"] is True
    assert metadata["Post-COVID Recovery"]["small_sample"] is False
    assert rankings.shape[1] == 6


def test_global_and_per_regime_shap_use_the_same_explainer_output_for_a_single_row():
    """A regime containing every row should match the global explanation exactly."""
    rng = np.random.default_rng(1)
    n = 15
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["x", "y"])
    y = X["x"] + rng.normal(scale=0.1, size=n)
    model = XGBForecastingModel(max_depth=2, n_estimators=10).fit(X, y)
    regimes = pd.Series(["only_regime"] * n)

    global_explanation = compute_shap_values(model, X)
    per_regime = compute_per_regime_shap(model, X, regimes)

    np.testing.assert_array_equal(global_explanation.values, per_regime["only_regime"].values)
