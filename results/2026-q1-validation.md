# 2026 Q1 validation

Forecast: 0.4440%. ONS actual (2026 Q1, real, CVM, SA, QoQ): 0.6000%.
Error: -0.1560 percentage points (absolute error 0.1560).

## Context: XGBoost's usual error in Post-COVID Recovery

Q1 2026 falls in the Post-COVID Recovery regime, so this single forecast is read against XGBoost's typical error there in the Sprint 4 evaluation, rather than judged on its own.

| Scheme | n | RMSE | MAE | MASE | R2 |
|---|---|---|---|---|---|
| expanding_window | 18 | 0.509 | 0.422 | 0.367 | 0.034 |
| regime_aligned | 90 | 0.543 | 0.435 | 0.379 | -0.098 |

The Q1 2026 absolute error (0.1560) is 0.37 times the expanding-window MAE for this regime (0.422) and 0.31 times its RMSE (0.509). A single quarter cannot confirm a trend either way; this is one data point read against the regime's usual error range, not a claim that the model has improved or degraded.
