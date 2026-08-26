# LASSO comparison: does a feature-selection model find exploitable macro signal?

Confirmatory diagnostic, not a model change. LASSO is linear regression with an L1 penalty, which
drives some coefficients to exactly zero, so it selects features as it fits. If the macroeconomic
predictors carried usable forward-looking signal, a model built to seek and keep the useful ones
should both retain them and forecast well. This run tests that directly.

Setup mirrors the Ridge baseline exactly for comparability: frozen 2000 to 2025 dataset, the same
one-step-ahead shift (103 rows), a StandardScaler plus estimator pipeline, alpha tuned by
RandomizedSearchCV over 20 logspace values on the first 75 percent of the data with expanding-window
inner cross-validation and neg-RMSE scoring, then evaluated under the same 8-fold expanding-window
scheme as the main results with a fresh scaler and LASSO fit inside each fold on that fold's
training rows only. Tuned alpha was 0.0695.

## Accuracy comparison (expanding-window CV)

| Model | Features | RMSE | MAE |
|---|---|---|---|
| GDP-history-only XGBoost (ablation) | 5 | 2.343 | 1.835 |
| XGBoost (full) | 17 | 2.397 | 1.873 |
| LightGBM | 17 | 2.497 | 1.961 |
| Macro-only XGBoost (ablation) | 12 | 2.640 | 2.097 |
| Ridge | 17 | 2.756 | 2.225 |
| LASSO | 17 available | 3.947 | 3.329 |
| ARIMA | 1 | 4.835 | 3.849 |

LASSO forecasts worse than every model except ARIMA, and notably worse than Ridge (2.756), even
though the two differ only in the type of penalty (L1 against L2).

## Which features LASSO kept (full-data fit, tuned alpha)

LASSO kept 12 of 17 features with nonzero coefficients and did not collapse toward GDP history:

- Kept (12): govt_consumption_growth (-0.75, the largest coefficient of all), gdp_yoy,
  business_confidence, gdp_lag_4, gdp_lag_1, brent_oil, trade_balance, cpi_inflation, gdp_growth,
  bank_rate, unemployment_rate, gbp_usd_rate. That is 7 macro features alongside 5 GDP-history
  features.
- Zeroed (5): gfcf_growth, consumer_confidence, gdp_rolling_mean_4q,
  business_confidence_rolling_mean_4q, yield_curve_slope.

## Reading

The result is subtler than the earlier ablation and should be read carefully. LASSO does assign
substantial in-sample weight to macro features, its single largest coefficient is a macro one
(govt_consumption_growth), so a linear feature-selector left to fit thinks the macro features are
useful. But those macro coefficients buy no out-of-sample accuracy: LASSO is the worst forecaster
of the whole set except ARIMA. The macro signal it latches onto is in-sample fit that does not
generalise to held-out quarters. XGBoost ignores the macro features and forecasts well on GDP
history alone; LASSO uses the macro features and is punished for it out-of-sample. Both point to
the same conclusion from opposite directions: the macro features carry in-sample fit but no
exploitable out-of-sample signal at this sample size (103 training rows).

Honest caveat: the GDP-history features are highly correlated with each other, and LASSO is known
to be unstable in which member of a correlated group it retains, it tends to pick one somewhat
arbitrarily and can zero a near-duplicate (here it zeroed gdp_rolling_mean_4q while keeping gdp_yoy).
So the exact selected and zeroed lists above should not be over-interpreted feature by feature. The
robust, reportable result is the poor out-of-sample RMSE, not the precise membership of the kept
set.

This is the fourth independent line of evidence that the macroeconomic predictors add no reliable
out-of-sample signal at n=103, alongside the SHAP zero-importance result, the capacity experiment,
and the feature ablation. It is the first to confirm the finding from the use-them direction (a
model actively fitting the macro features generalises worse) rather than the ignore-them direction
(XGBoost gives them zero importance and loses nothing).
