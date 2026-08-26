# 2026 Q2 forecast

XGBoost refit on the frozen 2000-2025 history (Sprint 3 cached hyperparameters, not retuned) and forecast from a constructed 2026 Q1 feature row. Under the one-step-ahead mapping the 2026 Q1 row predicts 2026 Q2. Training stayed at 2000-2025; the 2026 data is predictor input only.

## How the 2026 Q1 feature row was built

The GDP and confidence history that feeds the lag and rolling features was taken from the frozen dataset (training vintage), the real 2026 Q1 predictors were aggregated from the refreshed raw data, and the contemporaneous gdp_growth is the published 2026 Q1 actual (0.6 percent). The row was assembled by appending a 2026 Q1 row to the frozen raw columns and running the pipeline's engineer_features, so the engineered features match training.

## GDP-history features on the 2026 Q1 row (frozen vintage)

| Feature | Value | Source quarter | Vintage note |
|---|---|---|---|
| gdp_growth (contemporaneous) | 0.6000 | 2026 Q1 | published actual (0.6) |
| gdp_lag_1 | 0.1000 | 2025 Q4 | frozen |
| gdp_lag_4 | 0.7000 | 2025 Q1 | frozen 0.7, raw now revised to 0.6 |
| gdp_rolling_mean_4q | 0.2500 | 2025 Q2 to 2026 Q1 | frozen 2025, new 2026 Q1 |
| gdp_yoy | 1.0029 | 2025 Q2 to 2026 Q1 | frozen 2025, new 2026 Q1 |

## Real 2026 Q1 predictors and derived features

| Feature | Value |
|---|---|
| unemployment_rate | 4.9333 |
| cpi_inflation | 3.1000 |
| trade_balance | -4564.3333 |
| gfcf_growth | 0.3814 |
| govt_consumption_growth | 1.3064 |
| bank_rate | 3.7500 |
| gbp_usd_rate | 1.3188 |
| brent_oil | 80.7198 |
| business_confidence | -9.8900 |
| consumer_confidence | -12.4167 |
| business_confidence_rolling_mean_4q | -12.6944 |
| yield_curve_slope | 0.6612 |

## Forecast

**2026 Q2 gdp_growth forecast: 0.4440%**

This is the same value as the Q1 2026 forecast, and that is expected rather than a coincidence or a duplicated row. The two forecast rows are genuinely different and their two live features differ: the Q1 row (built from 2025 Q4) has gdp_growth 0.1 and gdp_lag_4 0.3, while the Q2 row (built from 2026 Q1) has gdp_growth 0.6 and gdp_lag_4 0.7. They still return the same forecast because the model is a coarse step function: it produces only seven distinct predicted values across the 103 training rows, and 0.4440 (the modal prediction, 87 of 103 rows, close to the sample mean of 0.42) is returned for any roughly normal-positive recent growth. Sweeping gdp_growth on the Q2 row shows the model returns 0.4440 for every value from about 0.1 up through 3.0 and only changes for near-zero or negative growth. Both forecast rows fall in the range where the model returns its most common prediction (0.444), so their different feature values map to the same output. The model has in effect learned a threshold rule and cannot distinguish a 0.1 percent quarter from a 0.6 percent quarter. This is the same near-null behaviour seen in the Sprint 4 evaluation, where the model barely improves on predicting the mean and R-squared is negative; it is reported here as the honest outcome, not smoothed over.

Not validated: ONS has not published 2026 Q2 GDP yet (next release 13 August 2026). The forecast will be validated against the published actual then, the same way as Q1.
