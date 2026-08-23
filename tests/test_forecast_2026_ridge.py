"""Tests for scripts/forecast_2026_ridge.py.

Only refit_ridge() is exercised here since it is fast and deterministic
given the cached hyperparameters; the full forecast comparison is
covered by the script's own main() (it logs both quarters' forecasts
against the ONS actuals).
"""

from __future__ import annotations

from scripts.forecast_2026_ridge import refit_ridge


def test_refit_ridge_returns_fitted_model_with_predict():
    """refit_ridge returns an object with a predict method, fit on the full frozen dataset."""
    model, best_params, dataset_hash, n_rows = refit_ridge()
    assert hasattr(model, "predict")
    assert "ridge__alpha" in best_params
    assert isinstance(dataset_hash, str) and dataset_hash
    assert n_rows == 103
