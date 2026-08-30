# LASSO comparison: does a feature-selection model find exploitable macro signal?

Confirmatory diagnostic, not a model change. LASSO is linear regression with an L1 penalty, which drives some coefficients to exactly zero, so it selects features as it fits. If the macroeconomic predictors carried usable forward-looking signal, a model built to seek and keep the useful ones should both retain them and forecast well.

Setup mirrors the Ridge baseline for comparability: frozen 2000-2025 dataset, the same one-step-ahead shift, a StandardScaler-plus-estimator pipeline, alpha tuned by RandomizedSearchCV over 20 logspace values on the first 75% of the data with expanding-window inner cross-validation and neg-RMSE scoring, then evaluated under the same 8-fold expanding-window scheme as the main results with a fresh scaler and LASSO fit inside each fold on that fold's training rows only. Tuned alpha was 0.0695.

## Accuracy (expanding-window CV)

| Model | RMSE | MAE |
|---|---|---|
| LASSO | 3.947 | 3.329 |

For the full multi-model comparison (XGBoost, Ridge, LightGBM, ARIMA, and the feature ablation variants), see results/feature-ablation.md and results/metrics/aggregated.csv.

## Which features LASSO kept (full-data fit, tuned alpha)

LASSO kept 12 of 17 features with nonzero coefficients (4 GDP-history, 8 macro):

- Kept (12): govt_consumption_growth (-0.75), gdp_yoy (-0.50), business_confidence (0.44), gdp_lag_4 (-0.33), gdp_lag_1 (-0.33), brent_oil (-0.21), trade_balance (0.16), cpi_inflation (-0.12), gdp_growth (-0.10), bank_rate (0.05), unemployment_rate (0.01), gbp_usd_rate (0.01).
- Zeroed (5): consumer_confidence, gdp_rolling_mean_4q, gfcf_growth, business_confidence_rolling_mean_4q, yield_curve_slope.

## Reading

LASSO assigns substantial in-sample weight to macro features — its single largest coefficient is a macro one (govt_consumption_growth, -0.75) — so a linear feature-selector left to fit thinks the macro features are useful. But those macro coefficients buy no out-of-sample accuracy: LASSO is the worst forecaster of every model tested except ARIMA. The macro signal it latches onto is in-sample fit that does not generalise to held-out quarters. XGBoost ignores the macro features and forecasts well on GDP history alone; LASSO uses the macro features and is punished for it out-of-sample. Both point to the same conclusion from opposite directions: the macro features carry in-sample fit but no exploitable out-of-sample signal at this sample size (103 training rows).

Honest caveat: the GDP-history features are highly correlated with each other, and LASSO is known to be unstable in which member of a correlated group it retains — it tends to pick one somewhat arbitrarily and can zero a near-duplicate (here it zeroed gdp_rolling_mean_4q while keeping gdp_yoy). So the exact selected and zeroed lists above should not be over-interpreted feature by feature. The robust, reportable result is the poor out-of-sample RMSE, not the precise membership of the kept set.
