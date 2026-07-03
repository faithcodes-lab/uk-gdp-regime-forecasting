# 2026 predictor coverage report

Checkpoint 1 of the 2026 forecast validation epic. Generated from raw CSVs refreshed via make download after fixing the hardcoded BoE Dateto=31/Dec/2025 cutoff in config/pipeline.yaml. Read-only: data/processed/final_dataset.parquet (the frozen 2000-2025, 104-quarter training set) is untouched by this report.

## Raw predictor coverage by quarter

| Predictor | Source | Frequency | Quarter | Obs count | Last obs date | Last value | Status |
|---|---|---|---|---|---|---|---|
| gdp_growth | ons | quarterly | 2026 Q1 | 1 | 2026-03-31 | 0.6 | complete |
| gdp_growth | ons | quarterly | 2026 Q2 | 0 | - | - | MISSING |
| unemployment_rate | ons | monthly | 2026 Q1 | 3 | 2026-03-01 | 4.9 | complete |
| unemployment_rate | ons | monthly | 2026 Q2 | 0 | - | - | MISSING |
| cpi_inflation | ons | monthly | 2026 Q1 | 3 | 2026-03-01 | 3.3 | complete |
| cpi_inflation | ons | monthly | 2026 Q2 | 2 | 2026-05-01 | 2.8 | PARTIAL (2/3 expected obs) |
| trade_balance | ons | monthly | 2026 Q1 | 3 | 2026-03-01 | -9658 | complete |
| trade_balance | ons | monthly | 2026 Q2 | 1 | 2026-04-01 | -8435 | PARTIAL (1/3 expected obs) |
| gfcf_growth | ons | quarterly | 2026 Q1 | 1 | 2026-03-31 | 1.39e+05 | complete |
| gfcf_growth | ons | quarterly | 2026 Q2 | 0 | - | - | MISSING |
| govt_consumption_growth | ons | quarterly | 2026 Q1 | 1 | 2026-03-31 | 1.511e+05 | complete |
| govt_consumption_growth | ons | quarterly | 2026 Q2 | 0 | - | - | MISSING |
| bank_rate | boe | monthly | 2026 Q1 | 63 | 2026-03-31 | 3.75 | complete |
| bank_rate | boe | monthly | 2026 Q2 | 61 | 2026-06-30 | 3.75 | complete |
| gbp_usd_rate | boe | monthly | 2026 Q1 | 63 | 2026-03-31 | 1.319 | complete |
| gbp_usd_rate | boe | monthly | 2026 Q2 | 61 | 2026-06-30 | 1.327 | complete |
| brent_oil | fred | daily | 2026 Q1 | 63 | 2026-03-31 | 126.7 | present (n=63, daily series) |
| brent_oil | fred | daily | 2026 Q2 | 60 | 2026-06-29 | 71.59 | present (n=60, daily series) |
| business_confidence | fred | monthly | 2026 Q1 | 3 | 2026-03-01 | -8.064 | complete |
| business_confidence | fred | monthly | 2026 Q2 | 2 | 2026-05-01 | -14.11 | PARTIAL (2/3 expected obs) |
| consumer_confidence | fred | monthly | 2026 Q1 | 3 | 2026-03-01 | -15.25 | complete |
| consumer_confidence | fred | monthly | 2026 Q2 | 2 | 2026-05-01 | -16.75 | PARTIAL (2/3 expected obs) |
| gilt_2y_yield | boe_yc | monthly | 2026 Q1 | 3 | 2026-03-31 | 4.283 | complete |
| gilt_2y_yield | boe_yc | monthly | 2026 Q2 | 2 | 2026-05-29 | 4.09 | PARTIAL (2/3 expected obs) |
| gilt_10y_yield | boe_yc | monthly | 2026 Q1 | 3 | 2026-03-31 | 4.944 | complete |
| gilt_10y_yield | boe_yc | monthly | 2026 Q2 | 2 | 2026-05-29 | 4.86 | PARTIAL (2/3 expected obs) |

## Derived: yield curve slope (10Y minus 2Y)

- **2026 Q1**: yield_curve_slope = 0.661 (10Y=4.944, 2Y=4.283, both present)
- **2026 Q2**: yield_curve_slope = 0.770 (10Y=4.860, 2Y=4.090, both present)

## GDP target vintage check

### GDP vintage drift (frozen training set vs. freshly re-downloaded raw)

ONS has revised the following quarters since the frozen 104-quarter training set was built. data/processed/final_dataset.parquet is untouched and still reflects the original vintage, which is correct, the training set stays frozen. Flagging this because any lag feature built for the 2026 forecast (gdp_lag_1, gdp_lag_4) must be sourced from the frozen vintage below, not the revised raw CSV, to stay consistent with what XGBoost was trained on.

| Quarter | Frozen (training vintage) | Current raw (revised) |
|---|---|---|
| 2024-03-31 | 0.8 | 0.7 |
| 2024-12-31 | 0.3 | 0.4 |
| 2025-03-31 | 0.7 | 0.6 |

2026 Q1 actual now present in the refreshed raw series: 0.6% (IHYQ / QNA vintage, real, CVM, SA, QoQ, see checkpoint 2 for which release this is).

## Summary

- **2026 Q1**: 13/13 predictors complete.
- **2026 Q2**: 3/13 predictors complete. Missing: gdp_growth, unemployment_rate, gfcf_growth, govt_consumption_growth. Partial: cpi_inflation, trade_balance, business_confidence, consumer_confidence, gilt_2y_yield, gilt_10y_yield.

