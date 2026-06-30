"""ARIMA univariate baseline in the shared ForecastingModel interface.

Defaults to ARIMA(1, 0, 0), the published macroeconomic baseline. Passing
order=None triggers a manual AIC grid search via select_arima_order, the
in-house substitute for pmdarima.auto_arima (unavailable on Python 3.13).
Convergence failures are caught and logged: lbfgs first, then powell as a
retry, then ARIMA(1, 0, 0) as the final fallback. statsmodels ARIMA is
deterministic for a given series and order, so no random_state is wired
in.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.exceptions import NotFittedError
from statsmodels.tsa.arima.model import ARIMA as _StatsmodelsARIMA

from src.models.interface import ForecastingModel

_FALLBACK_ORDER: tuple[int, int, int] = (1, 0, 0)


def select_arima_order(
    y: pd.Series | np.ndarray,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
) -> tuple[int, int, int]:
    """Picks (p, d, q) by AIC over a small grid. Substitute for pmdarima.auto_arima.

    Returns the order with the lowest AIC across the grid. Orders that
    fail to fit are skipped silently. Falls back to (1, 0, 0) if every
    order in the grid fails.
    """
    best_order = _FALLBACK_ORDER
    best_aic = float("inf")
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    fit = _StatsmodelsARIMA(y, order=(p, d, q)).fit()
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_order = (p, d, q)
                except Exception:
                    # Skip orders that fail to fit; common for short series
                    # or near-non-stationary data. Continue the grid search.
                    continue
    return best_order


def _fit_with_fallback(
    y: pd.Series | np.ndarray,
    order: tuple[int, int, int],
) -> tuple[Any, tuple[int, int, int]]:
    """Fits ARIMA(y, order) with lbfgs, retries with powell, then falls back to ARIMA(1, 0, 0).

    Returns (fitted_results, used_order). Each substitution is logged via
    loguru so the audit trail survives in the log file.
    """
    try:
        fitted = _StatsmodelsARIMA(y, order=order).fit(method="lbfgs")
        return fitted, order
    except Exception as exc_lbfgs:
        logger.warning(f"ARIMA{order} failed with lbfgs ({exc_lbfgs}); retrying with powell")
        try:
            fitted = _StatsmodelsARIMA(y, order=order).fit(method="powell")
            return fitted, order
        except Exception as exc_powell:
            logger.warning(
                f"ARIMA{order} failed with powell ({exc_powell}); "
                f"falling back to ARIMA{_FALLBACK_ORDER}"
            )
            fitted = _StatsmodelsARIMA(y, order=_FALLBACK_ORDER).fit()
            return fitted, _FALLBACK_ORDER


class ARIMAModel(ForecastingModel):
    """ARIMA in the shared ForecastingModel interface.

    Thin wrapper around statsmodels ARIMA. fit ignores X (ARIMA models
    only the target series); predict returns a static multi-step forecast
    from the fit-time state; predict_one_step_ahead refits per call and
    is the unit operation used by cross_validate_arima.
    """

    def __init__(
        self,
        order: tuple[int, int, int] | None = (1, 0, 0),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self._fitted: Any = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ARIMAModel":
        """Fits ARIMA on y. X is ignored, accepted only for interface compatibility."""
        if self.order is None:
            self.order = select_arima_order(y)
        self._fitted, self.order = _fit_with_fallback(y, self.order)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns len(X) multi-step forecasts from the fit-time state. X is ignored.

        For one-step-ahead-with-refit CV use cross_validate_arima instead.
        """
        if self._fitted is None:
            raise NotFittedError("ARIMAModel has not been fit yet")
        forecast = self._fitted.forecast(steps=len(X))
        return np.asarray(forecast)

    def predict_one_step_ahead(self, history_y: pd.Series | np.ndarray) -> float:
        """Refits ARIMA on history_y and returns the next single forecast value."""
        fitted, _ = _fit_with_fallback(history_y, self.order)
        return float(np.asarray(fitted.forecast(steps=1))[0])

    def get_params(self) -> dict[str, Any]:
        """Returns the order and seasonal_order as a dict."""
        return {"order": self.order, "seasonal_order": self.seasonal_order}
