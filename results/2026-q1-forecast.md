# 2026 Q1 forecast

XGBoost refit on the frozen 2000-2025 history (Sprint 3 cached hyperparameters, not retuned) and forecast from the 2025 Q4 row already in that frozen dataset. No 2026 data was used as model input; training stayed at 2000-2025.

Training rows: 103. Dataset hash: 9b72759071dd458fb171a2ca6a778952.

## Lag and rolling features used (2025 Q4 row, frozen vintage)

| Feature | Value | Source quarter | Vintage note |
|---|---|---|---|
| gdp_growth (contemporaneous) | 0.1 | 2025 Q4 | unrevised |
| gdp_lag_1 | 0.1 | 2025 Q3 | unrevised |
| gdp_lag_4 | 0.3 | 2024 Q4 | frozen 0.3, raw now revised to 0.4 |
| gdp_rolling_mean_4q | 0.27499999999999936 | 2025 Q1 to Q4 | window includes 2025 Q1, frozen 0.7, raw now revised to 0.6 |
| gdp_yoy | 1.1033037013999714 | 2025 Q1 to Q4 | same window, same drift note |

All five values were read directly from data/processed/final_dataset.parquet, so the drift-affected quarters (2024 Q4 and 2025 Q1) are the frozen training-vintage values, not the freshly re-downloaded raw ones.

## Forecast

**2026 Q1 gdp_growth forecast: 0.4440%**

Not yet validated against the ONS actual (0.6%). That is the next step.
