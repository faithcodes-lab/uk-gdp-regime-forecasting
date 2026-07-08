"""Tests for src/explainability/stability.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.explainability.stability import (
    bootstrap_rankings,
    bootstrap_spearman_ci,
    classify_stability,
    pairwise_spearman_matrix,
)
from src.models.xgboost_model import XGBForecastingModel


def test_pairwise_spearman_identical_rankings_give_rho_one():
    rankings = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]}, index=["f1", "f2", "f3", "f4"])
    matrix = pairwise_spearman_matrix(rankings)
    assert matrix.loc["a", "b"] == pytest.approx(1.0)
    assert matrix.loc["a", "a"] == pytest.approx(1.0)


def test_pairwise_spearman_reversed_rankings_give_rho_minus_one():
    rankings = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]}, index=["f1", "f2", "f3", "f4"])
    matrix = pairwise_spearman_matrix(rankings)
    assert matrix.loc["a", "b"] == pytest.approx(-1.0)


def test_pairwise_spearman_unrelated_rankings_give_rho_near_zero():
    rankings = pd.DataFrame(
        {"a": [1, 2, 3, 4, 5, 6], "b": [3, 6, 1, 5, 2, 4]},
        index=["f1", "f2", "f3", "f4", "f5", "f6"],
    )
    matrix = pairwise_spearman_matrix(rankings)
    assert abs(matrix.loc["a", "b"]) < 0.5


def test_pairwise_spearman_matrix_is_symmetric_and_square():
    rankings = pd.DataFrame(
        {"x": [1, 2, 3], "y": [2, 1, 3], "z": [3, 2, 1]}, index=["f1", "f2", "f3"]
    )
    matrix = pairwise_spearman_matrix(rankings)
    assert matrix.shape == (3, 3)
    np.testing.assert_allclose(matrix.to_numpy(), matrix.to_numpy().T)


@pytest.mark.parametrize(
    "rho,expected",
    [
        (0.9, "stable"),
        (0.61, "stable"),
        (0.6, "moderately stable"),
        (0.45, "moderately stable"),
        (0.31, "moderately stable"),
        (0.3, "unstable"),
        (0.0, "unstable"),
        (-0.5, "unstable"),
    ],
)
def test_classify_stability_bands(rho, expected):
    assert classify_stability(rho) == expected


@pytest.fixture
def small_regime_model_and_data() -> tuple[XGBForecastingModel, pd.DataFrame]:
    rng = np.random.default_rng(42)
    n = 6
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = X["a"] * 2 - X["b"] + rng.normal(scale=0.1, size=n)
    model = XGBForecastingModel(max_depth=2, n_estimators=20).fit(X, y)
    return model, X


def test_bootstrap_rankings_shape(small_regime_model_and_data):
    model, X = small_regime_model_and_data
    result = bootstrap_rankings(model, X, n_bootstrap=50, random_state=42)
    assert result.shape == (50, 3)


def test_bootstrap_rankings_deterministic_with_seed(small_regime_model_and_data):
    model, X = small_regime_model_and_data
    first = bootstrap_rankings(model, X, n_bootstrap=50, random_state=42)
    second = bootstrap_rankings(model, X, n_bootstrap=50, random_state=42)
    np.testing.assert_array_equal(first, second)


def test_bootstrap_rankings_differs_with_different_seed():
    # A 6-row, one-dominant-feature fixture (as used elsewhere in this file, mirroring the
    # real GFC/COVID regime size) gives XGBoost so little to split on that two features get
    # an exact-zero SHAP value on every row, tying regardless of resampling composition. That
    # is a genuine small-sample finding, not a resampling bug, so it is the wrong fixture to
    # prove the seed changes anything. This test uses a larger, multi-feature-dependent
    # fixture instead, specifically to confirm the random_state argument is wired through.
    rng = np.random.default_rng(3)
    n = 30
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = X["a"] - X["b"] + 0.5 * X["c"] + rng.normal(scale=0.1, size=n)
    model = XGBForecastingModel(max_depth=3, n_estimators=30).fit(X, y)

    first = bootstrap_rankings(model, X, n_bootstrap=50, random_state=42)
    second = bootstrap_rankings(model, X, n_bootstrap=50, random_state=7)
    assert not np.array_equal(first, second)


def test_bootstrap_spearman_ci_is_one_one_for_identical_distributions(small_regime_model_and_data):
    model, X = small_regime_model_and_data
    rankings = bootstrap_rankings(model, X, n_bootstrap=50, random_state=42)
    lower, upper = bootstrap_spearman_ci(rankings, rankings, n_bootstrap=50)
    assert lower == pytest.approx(1.0)
    assert upper == pytest.approx(1.0)


def test_bootstrap_spearman_ci_returns_lower_le_upper(small_regime_model_and_data):
    model, X = small_regime_model_and_data
    rankings_a = bootstrap_rankings(model, X, n_bootstrap=50, random_state=42)
    rankings_b = bootstrap_rankings(model, X, n_bootstrap=50, random_state=7)
    lower, upper = bootstrap_spearman_ci(rankings_a, rankings_b, n_bootstrap=50)
    assert lower <= upper
    assert -1.0 <= lower <= 1.0
    assert -1.0 <= upper <= 1.0
