"""Tests for src/models/arima.py.

The two tests that matter most are test_arima_recovers_ar1_coefficient
(proves the underlying fit actually works) and
test_arima_fallback_on_convergence_failure_is_logged (proves substitutions
are logged, not silent).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from loguru import logger as loguru_logger
from sklearn.exceptions import NotFittedError

from src.models import arima as arima_module
from src.models.arima import ARIMAModel, select_arima_order
from src.models.interface import ForecastingModel


def _ar1_series(n: int = 200, phi: float = 0.7, noise: float = 0.1, seed: int = 42) -> pd.Series:
    """Generates a synthetic AR(1) series with the given phi coefficient."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.normal(scale=noise)
    return pd.Series(y)


def _dummy_X(n: int) -> pd.DataFrame:
    """Builds a placeholder X DataFrame to pass to ARIMA's fit signature."""
    return pd.DataFrame({"f1": np.arange(n, dtype=float)})


def test_arima_subclasses_forecasting_model():
    """ARIMAModel is a subclass of ForecastingModel."""
    assert issubclass(ARIMAModel, ForecastingModel)


def test_arima_get_params_returns_order_and_seasonal_order():
    """get_params returns the order and seasonal_order as a dict."""
    model = ARIMAModel(order=(2, 1, 1), seasonal_order=(1, 0, 0, 4))
    assert model.get_params() == {"order": (2, 1, 1), "seasonal_order": (1, 0, 0, 4)}


def test_arima_fit_returns_self():
    """fit returns the model itself so calls can be chained."""
    y = _ar1_series(n=80)
    model = ARIMAModel(order=(1, 0, 0))
    returned = model.fit(_dummy_X(80), y)
    assert returned is model


def test_arima_fit_ignores_X():
    """Passing different X values produces identical fitted parameters."""
    y = _ar1_series(n=80)
    X_a = pd.DataFrame({"unrelated": np.zeros(80)})
    X_b = pd.DataFrame({"a": np.arange(80, dtype=float), "b": np.arange(80, dtype=float) ** 2})

    model_a = ARIMAModel(order=(1, 0, 0)).fit(X_a, y)
    model_b = ARIMAModel(order=(1, 0, 0)).fit(X_b, y)

    np.testing.assert_array_almost_equal(
        model_a._fitted.params.to_numpy(), model_b._fitted.params.to_numpy()
    )


def test_arima_predict_returns_correct_shape():
    """predict returns one value per row of X (the test window length)."""
    y = _ar1_series(n=80)
    model = ARIMAModel(order=(1, 0, 0)).fit(_dummy_X(80), y)
    preds = model.predict(_dummy_X(4))
    assert preds.shape == (4,)


def test_arima_recovers_ar1_coefficient_on_synthetic_data():
    """ARIMA(1,0,0) on a 500-point AR(1) series recovers the true phi within 0.05."""
    phi_true = 0.7
    y = _ar1_series(n=500, phi=phi_true, noise=0.1, seed=42)

    model = ARIMAModel(order=(1, 0, 0)).fit(_dummy_X(500), y)
    ar_coef = float(model._fitted.arparams[0])

    assert (
        abs(ar_coef - phi_true) < 0.05
    ), f"AR coefficient {ar_coef:.4f} not within 0.05 of true phi {phi_true}"


def test_arima_predict_one_step_ahead_returns_scalar():
    """predict_one_step_ahead returns a single float."""
    y = _ar1_series(n=80)
    model = ARIMAModel(order=(1, 0, 0))
    pred = model.predict_one_step_ahead(y)
    assert isinstance(pred, float)


def test_arima_deterministic_on_same_data():
    """Two ARIMAModel fits on the same data and order produce identical predictions."""
    y = _ar1_series(n=80)
    a = ARIMAModel(order=(1, 0, 0)).fit(_dummy_X(80), y)
    b = ARIMAModel(order=(1, 0, 0)).fit(_dummy_X(80), y)
    np.testing.assert_array_almost_equal(a.predict(_dummy_X(4)), b.predict(_dummy_X(4)))


def test_arima_fallback_on_convergence_failure_is_logged(monkeypatch):
    """If lbfgs and powell both fail, fallback uses ARIMA(1,0,0) and logs each substitution."""
    captured: list[str] = []
    sink_id = loguru_logger.add(captured.append, format="{message}", level="WARNING")
    try:
        bad_order = (5, 2, 5)
        good_order = (1, 0, 0)

        original_arima_cls = arima_module._StatsmodelsARIMA

        def patched_arima(y, order=None, **kwargs):
            obj = original_arima_cls(y, order=order, **kwargs)
            original_fit = obj.fit

            def patched_fit(*args, **kwargs):
                # Fail for bad_order; succeed for the fallback (1,0,0) so we
                # can verify the fallback path actually completes.
                if order == bad_order:
                    raise ValueError("synthetic convergence failure")
                return original_fit(*args, **kwargs)

            obj.fit = patched_fit
            return obj

        monkeypatch.setattr(arima_module, "_StatsmodelsARIMA", patched_arima)

        y = _ar1_series(n=80)
        _fitted, used_order = arima_module._fit_with_fallback(y, bad_order)

        assert used_order == good_order
        assert any(
            "powell" in m.lower() for m in captured
        ), f"Expected default-to-powell warning in logs, got: {captured}"
        assert any(
            "falling back" in m.lower() and "(1, 0, 0)" in m for m in captured
        ), f"Expected fallback-to-(1,0,0) warning in logs, got: {captured}"
    finally:
        loguru_logger.remove(sink_id)


def test_arima_fit_actually_uses_requested_order_not_fallback():
    """Fitting ARIMA(3, 0, 0) against real statsmodels uses (3, 0, 0), not the (1, 0, 0) fallback.

    Guards against the API misuse where .fit(method="lbfgs") was treated as
    an estimator name by statsmodels and always raised, so the fallback to
    ARIMA(1, 0, 0) fired silently even on valid orders. This test runs
    against real statsmodels (no mock) so the fallback only fires if the
    real fit call genuinely fails.
    """
    y = _ar1_series(n=500, phi=0.7, noise=0.1, seed=42)
    _fitted, used_order = arima_module._fit_with_fallback(y, order=(3, 0, 0))
    assert used_order == (
        3,
        0,
        0,
    ), f"expected ARIMA(3, 0, 0) to fit on AR(1) data, got fallback {used_order}"


def test_arima_predict_before_fit_raises():
    """Calling predict before fit raises NotFittedError."""
    model = ARIMAModel(order=(1, 0, 0))
    with pytest.raises(NotFittedError):
        model.predict(_dummy_X(4))


def test_select_arima_order_returns_valid_tuple():
    """select_arima_order returns a (p, d, q) tuple within the given grid bounds."""
    # small grid for test speed; the order returned must be inside the bounds
    y = _ar1_series(n=80)
    order = select_arima_order(y, max_p=1, max_d=1, max_q=1)
    p, d, q = order
    assert 0 <= p <= 1
    assert 0 <= d <= 1
    assert 0 <= q <= 1


def test_arima_with_order_none_uses_select_arima_order(monkeypatch):
    """Passing order=None triggers select_arima_order during fit."""
    # mock select_arima_order so this test does not depend on grid-search timing
    called_with: list[pd.Series] = []

    def mock_select(y, **kwargs):
        called_with.append(y)
        return (1, 0, 0)

    monkeypatch.setattr(arima_module, "select_arima_order", mock_select)

    y = _ar1_series(n=100)
    model = ARIMAModel(order=None).fit(_dummy_X(100), y)

    assert len(called_with) == 1
    assert model.order == (1, 0, 0)
