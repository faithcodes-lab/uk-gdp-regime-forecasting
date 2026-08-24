# How sensitive are the crisis-regime results to individual quarters?

Confirmatory diagnostic, not a model change. Leave-one-quarter-out RMSE for each crisis regime (Global Financial Crisis, COVID-19 Shock), model, and held-out quarter under regime-aligned CV, each with only 6 quarters.

## Largest swing per regime

| Regime | Largest |swing| across all models and quarters |
|---|---|
| COVID-19 Shock | 52.2% |
| Global Financial Crisis | 11.9% |

Single largest swing overall: arima in COVID-19 Shock, dropping 2020-09-30 moves RMSE from 25.106 to 11.993 (-52.2%).

## Full results

| Model | Regime | Held-out quarter | Full RMSE | Leave-one-out RMSE | Swing |
|---|---|---|---|---|---|
| arima | COVID-19 Shock | 2020-03-31 | 25.106 | 27.470 | +9.4% |
| arima | COVID-19 Shock | 2020-06-30 | 25.106 | 26.356 | +5.0% |
| arima | COVID-19 Shock | 2020-09-30 | 25.106 | 11.993 | -52.2% |
| arima | COVID-19 Shock | 2020-12-31 | 25.106 | 26.505 | +5.6% |
| arima | COVID-19 Shock | 2021-03-31 | 25.106 | 27.344 | +8.9% |
| arima | COVID-19 Shock | 2021-06-30 | 25.106 | 27.177 | +8.2% |
| arima | Global Financial Crisis | 2008-06-30 | 0.965 | 0.976 | +1.1% |
| arima | Global Financial Crisis | 2008-09-30 | 0.965 | 0.853 | -11.5% |
| arima | Global Financial Crisis | 2008-12-31 | 0.965 | 1.040 | +7.8% |
| arima | Global Financial Crisis | 2009-03-31 | 0.965 | 1.057 | +9.5% |
| arima | Global Financial Crisis | 2009-06-30 | 0.965 | 0.928 | -3.8% |
| arima | Global Financial Crisis | 2009-09-30 | 0.965 | 0.919 | -4.8% |
| lightgbm | COVID-19 Shock | 2020-03-31 | 11.142 | 12.131 | +8.9% |
| lightgbm | COVID-19 Shock | 2020-06-30 | 11.142 | 8.254 | -25.9% |
| lightgbm | COVID-19 Shock | 2020-09-30 | 11.142 | 9.612 | -13.7% |
| lightgbm | COVID-19 Shock | 2020-12-31 | 11.142 | 12.197 | +9.5% |
| lightgbm | COVID-19 Shock | 2021-03-31 | 11.142 | 12.183 | +9.4% |
| lightgbm | COVID-19 Shock | 2021-06-30 | 11.142 | 11.830 | +6.2% |
| lightgbm | Global Financial Crisis | 2008-06-30 | 1.872 | 1.988 | +6.2% |
| lightgbm | Global Financial Crisis | 2008-09-30 | 1.872 | 1.793 | -4.2% |
| lightgbm | Global Financial Crisis | 2008-12-31 | 1.872 | 1.650 | -11.9% |
| lightgbm | Global Financial Crisis | 2009-03-31 | 1.872 | 1.714 | -8.4% |
| lightgbm | Global Financial Crisis | 2009-06-30 | 1.872 | 2.011 | +7.4% |
| lightgbm | Global Financial Crisis | 2009-09-30 | 1.872 | 2.038 | +8.9% |
| ridge | COVID-19 Shock | 2020-03-31 | 11.896 | 12.958 | +8.9% |
| ridge | COVID-19 Shock | 2020-06-30 | 11.896 | 9.534 | -19.9% |
| ridge | COVID-19 Shock | 2020-09-30 | 11.896 | 9.518 | -20.0% |
| ridge | COVID-19 Shock | 2020-12-31 | 11.896 | 13.030 | +9.5% |
| ridge | COVID-19 Shock | 2021-03-31 | 11.896 | 12.985 | +9.2% |
| ridge | COVID-19 Shock | 2021-06-30 | 11.896 | 12.701 | +6.8% |
| ridge | Global Financial Crisis | 2008-06-30 | 1.797 | 1.903 | +5.9% |
| ridge | Global Financial Crisis | 2008-09-30 | 1.797 | 1.714 | -4.6% |
| ridge | Global Financial Crisis | 2008-12-31 | 1.797 | 1.594 | -11.3% |
| ridge | Global Financial Crisis | 2009-03-31 | 1.797 | 1.639 | -8.8% |
| ridge | Global Financial Crisis | 2009-06-30 | 1.797 | 1.940 | +8.0% |
| ridge | Global Financial Crisis | 2009-09-30 | 1.797 | 1.954 | +8.8% |
| xgboost | COVID-19 Shock | 2020-03-31 | 11.182 | 12.169 | +8.8% |
| xgboost | COVID-19 Shock | 2020-06-30 | 11.182 | 8.361 | -25.2% |
| xgboost | COVID-19 Shock | 2020-09-30 | 11.182 | 9.617 | -14.0% |
| xgboost | COVID-19 Shock | 2020-12-31 | 11.182 | 12.241 | +9.5% |
| xgboost | COVID-19 Shock | 2021-03-31 | 11.182 | 12.230 | +9.4% |
| xgboost | COVID-19 Shock | 2021-06-30 | 11.182 | 11.846 | +5.9% |
| xgboost | Global Financial Crisis | 2008-06-30 | 1.877 | 1.986 | +5.8% |
| xgboost | Global Financial Crisis | 2008-09-30 | 1.877 | 1.782 | -5.1% |
| xgboost | Global Financial Crisis | 2008-12-31 | 1.877 | 1.686 | -10.2% |
| xgboost | Global Financial Crisis | 2009-03-31 | 1.877 | 1.717 | -8.5% |
| xgboost | Global Financial Crisis | 2009-06-30 | 1.877 | 2.015 | +7.3% |
| xgboost | Global Financial Crisis | 2009-09-30 | 1.877 | 2.043 | +8.8% |
