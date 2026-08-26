# Feature ablation: confirming the two-feature finding

Confirmatory diagnostic, not a model change. Refits XGBoost on the frozen 2000 to 2025 dataset
with the Sprint 3 cached hyperparameters (50 trees, max depth 4, learning rate 0.01), under the
same expanding-window cross-validation as the main evaluation (8 folds, test size 4, a fresh model
fit inside each fold on that fold's training rows only). Training stays at 2000 to 2025. The point
is to test where the model's predictive power actually comes from by holding out whole groups of
features.

The full-feature baseline row reproduces the main evaluation's recorded expanding-window RMSE
(2.3974 here against the recorded 2.397391), which certifies this diagnostic uses the same data,
the same cross-validation, and the same model as the headline results, so the ablation rows are
directly comparable.

## Results

| Variant | Features used | Count | RMSE | MAE |
|---|---|---|---|---|
| Full feature set (baseline) | all 17 | 17 | 2.3974 | 1.8733 |
| GDP-history only | gdp_growth, gdp_lag_1, gdp_lag_4, gdp_rolling_mean_4q, gdp_yoy | 5 | 2.3430 | 1.8347 |
| Macro only | 10 raw predictors plus yield_curve_slope and business_confidence_rolling_mean_4q | 12 | 2.6402 | 2.0967 |

The 5 GDP-history features and the 12 macro features are an exact, non-overlapping partition of all
17 (checked in the diagnostic before it runs).

## Reading

GDP history alone matches, and slightly beats, the full model: dropping all 12 macro features
moved RMSE from 2.397 to 2.343 and MAE from 1.873 to 1.835, both small improvements rather than a
loss. The macro features were not carrying signal the full model needed. Macro alone, with no
access to GDP's own recent values, is clearly worse at RMSE 2.640, about 13% worse than
GDP-history-only.

This is a third independent line of evidence for the two-feature finding, alongside the SHAP
zero-importance result (the fitted model gives exactly zero SHAP to all 15 non-GDP-history
features in every regime) and the capacity experiment (adding capacity makes the model use more
features but worsens cross-validated RMSE). All three point the same way: GDP history carries the
forecasting signal, and the macroeconomic predictors add no exploitable out-of-sample information
at this sample size (103 training rows). It also unifies with the Sprint 4 near-null result, since
a model that is in substance a two-feature autoregression on GDP's own recent history is close to
the naive baseline it fails to beat.
