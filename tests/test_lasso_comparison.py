"""Tests for scripts/lasso_comparison.py.

Only the pipeline construction and coefficient-extraction logic is unit
tested here since they are fast and deterministic; the full alpha-tuning
and CV run is covered by the reproduction check in the script's own
main() (it logs the tuned alpha and RMSE/MAE against the recorded
values from results/lasso-comparison.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.lasso_comparison import _build_lasso_pipeline, _full_data_coefficients


def test_build_lasso_pipeline_has_scaler_and_lasso_steps():
    """The pipeline has exactly two named steps: scaler then lasso."""
    pipeline = _build_lasso_pipeline(alpha=0.5)
    assert list(pipeline.named_steps.keys()) == ["scaler", "lasso"]
    assert pipeline.named_steps["lasso"].alpha == 0.5


def test_full_data_coefficients_indexed_by_column_name():
    """Coefficients are returned as a Series indexed by the feature columns."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=50),
            "b": rng.normal(size=50),
        }
    )
    y = pd.Series(2.0 * X["a"] + rng.normal(scale=0.01, size=50))
    coefficients = _full_data_coefficients(X, y, alpha=0.01)
    assert set(coefficients.index) == {"a", "b"}


def test_full_data_coefficients_sorted_by_absolute_magnitude_descending():
    """The strongly predictive feature ('a') has the largest |coefficient|, listed first."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=50),
            "b": rng.normal(size=50),
        }
    )
    y = pd.Series(5.0 * X["a"] + rng.normal(scale=0.01, size=50))
    coefficients = _full_data_coefficients(X, y, alpha=0.01)
    assert coefficients.index[0] == "a"
    assert abs(coefficients.iloc[0]) >= abs(coefficients.iloc[1])


def test_lasso_alpha_zero_out_irrelevant_feature():
    """A high enough alpha drives an uninformative feature's coefficient to exactly zero."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "informative": rng.normal(size=50),
            "noise": rng.normal(size=50),
        }
    )
    y = pd.Series(3.0 * X["informative"] + rng.normal(scale=0.01, size=50))
    coefficients = _full_data_coefficients(X, y, alpha=1.0)
    assert coefficients["noise"] == 0.0
