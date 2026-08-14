# 2026 Q2 validation

Forecast: 0.4440%. ONS actual (2026 Q2, first quarterly estimate, QoQ): 0.4000%.
Error: 0.0440 percentage points (absolute error 0.0440).

## Context: XGBoost's usual error in Post-COVID Recovery

Q2 2026 falls in the Post-COVID Recovery regime, so this single forecast is read against XGBoost's typical error there in the Sprint 4 evaluation, rather than judged on its own.

| Scheme | n | RMSE | MAE | MASE | R2 |
|---|---|---|---|---|---|
| expanding_window | 18 | 0.509 | 0.422 | 0.367 | 0.034 |
| regime_aligned | 90 | 0.543 | 0.435 | 0.379 | -0.098 |

The Q2 2026 absolute error (0.0440) is 0.10 times the expanding-window MAE for this regime (0.422) and 0.09 times its RMSE (0.509), smaller than the Q1 2026 error (0.37x MAE, 0.31x RMSE). A single quarter cannot confirm a trend either way; this is one more data point read against the regime's usual error range, not a claim that the model has improved.

## Reading this alongside the Section 4.6 finding

Q2 2026 landing close to the model's near-constant prediction (0.444) is expected given that Q2 was itself an unremarkable, "normal" quarter. Per the ONS "GDP first quarterly estimate, UK: April to June 2026" bulletin (published 13 August 2026): "UK real gross domestic product (GDP) is estimated to have increased by 0.4% in Quarter 2 (Apr to June) 2026, following growth of 0.6% in Quarter 1 (Jan to Mar) 2026." On the sector breakdown: "In output terms, growth in the latest quarter was mainly caused by an increase of 0.5% in the services sector; the construction sector increased by 0.3% and production output showed no growth." This is consistent with, not a contradiction of, the model's insensitivity to ordinary quarter-to-quarter variation: a low-variance prediction lands close to reality when reality is also low-variance. It would not be expected to hold if Q2 had been another extreme quarter, the same way the model badly missed the COVID-19 Shock regime.
