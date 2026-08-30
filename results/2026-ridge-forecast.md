# 2026 Ridge forecast: second-model comparison

XGBoost was selected for the headline 2026 forecast on the lowest mean RMSE, MAE and MASE across both CV schemes, and as the only model with a positive R-squared anywhere (0.067, regime-aligned). But Section 4.2's Diebold-Mariano tests found no statistically reliable difference between the models, and Ridge had a better median RMSE and MASE. This report refits Ridge under identical conditions (same frozen 2000-2025 dataset, same one-step-ahead mapping, same cached hyperparameters, no retuning) and forecasts both 2026 quarters, so the two forecasts can be compared directly.

## Forecast comparison

| Model | Q1 2026 forecast | Q1 error | Q2 2026 forecast | Q2 error |
|---|---|---|---|---|
| Ridge | 0.2481% | -0.3519 | 0.0153% | -0.3847 |
| XGBoost | 0.4440% | -0.1560 | 0.4440% | +0.0440 |

ONS actuals: Q1 2026 0.6000%, Q2 2026 0.4000%.

## Reading

Ridge and XGBoost diverge more than Section 4.2's cross-validated near-tie might suggest: Ridge forecasts lower than XGBoost for both quarters, and lands further from the ONS actual both times. Two quarters is too small a sample to conclude Ridge is the worse model here, since 4.2's own tests found no reliable difference between them under cross-validation; this is one realisation, not a repeated comparison. What it does show is that the model choice was not inconsequential for these two forecasts, so presenting only XGBoost's number without this comparison would have understated how much the point forecast depends on which model was picked.
