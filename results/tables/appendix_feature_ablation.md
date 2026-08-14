**Feature ablation (XGBoost, expanding-window CV). XGBoost refit on three non-overlapping feature sets, using the same cached hyperparameters and cross-validation as the main evaluation.**

| Variant | Features used | Count | RMSE | MAE |
| --- | --- | --- | --- | --- |
| Full feature set (baseline) | all 17 | 17 | 2.3974 | 1.8733 |
| GDP-history only | gdp_growth, gdp_lag_1, gdp_lag_4, gdp_rolling_mean_4q, gdp_yoy | 5 | 2.3430 | 1.8347 |
| Macro only | 10 raw predictors plus yield_curve_slope and business_confidence_rolling_mean_4q | 12 | 2.6402 | 2.0967 |

The GDP-history and macro feature sets are an exact, non-overlapping partition of all 17 features. GDP-history-only matches and slightly beats the full model (RMSE 2.343 vs. 2.397), while macro-only is clearly worse (2.640), about 13% worse than GDP-history-only.
