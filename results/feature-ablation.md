# Feature ablation: does removing the macroeconomic predictors cost XGBoost anything?

Confirmatory diagnostic, not a model change. Refits XGBoost on the frozen 2000-2025 dataset under expanding-window CV with the same cached hyperparameters as the main evaluation, on three non-overlapping feature sets.

## Accuracy comparison (expanding-window CV, mean across 8 folds)

| Variant | Features | RMSE | MAE |
|---|---|---|---|
| Full feature set (baseline) | 17 | 2.3974 | 1.8733 |
| GDP-history only | 5 | 2.3430 | 1.8347 |
| Macro only | 12 | 2.6402 | 2.0967 |

GDP-history features: gdp_growth, gdp_lag_1, gdp_lag_4, gdp_rolling_mean_4q, gdp_yoy

Macro features: unemployment_rate, cpi_inflation, trade_balance, gfcf_growth, govt_consumption_growth, bank_rate, gbp_usd_rate, brent_oil, business_confidence, consumer_confidence, business_confidence_rolling_mean_4q, yield_curve_slope
