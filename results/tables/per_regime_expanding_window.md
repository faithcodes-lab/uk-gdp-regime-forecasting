**Per-regime forecast accuracy (expanding-window CV). Lower values indicate better performance for RMSE, MAE, and MASE; higher R2 indicates better fit. Asterisk marks small-sample regimes.**

| Model | Regime | n | RMSE | MAE | MASE | R2 |
| --- | --- | --- | --- | --- | --- | --- |
| arima | Brexit Transition* | 8 | 0.325 | 0.290 | 0.253 | -1.822 |
| arima | COVID-19 Shock* | 6 | 25.106 | 18.050 | 15.709 | -4.083 |
| arima | Post-COVID Recovery | 18 | 1.157 | 0.697 | 0.606 | -3.983 |
| lightgbm | Brexit Transition* | 8 | 0.335 | 0.276 | 0.241 | -1.984 |
| lightgbm | COVID-19 Shock* | 6 | 11.147 | 8.299 | 7.223 | -0.002 |
| lightgbm | Post-COVID Recovery | 18 | 0.835 | 0.597 | 0.520 | -1.595 |
| ridge | Brexit Transition* | 8 | 0.323 | 0.292 | 0.254 | -1.789 |
| ridge | COVID-19 Shock* | 6 | 12.045 | 8.342 | 7.260 | -0.170 |
| ridge | Post-COVID Recovery | 18 | 1.738 | 1.046 | 0.910 | -10.244 |
| xgboost | Brexit Transition* | 8 | 0.337 | 0.296 | 0.258 | -2.024 |
| xgboost | COVID-19 Shock* | 6 | 11.194 | 8.331 | 7.251 | -0.011 |
| xgboost | Post-COVID Recovery | 18 | 0.509 | 0.422 | 0.367 | 0.034 |

* Small sample (n < 10 quarters); bootstrap confidence intervals reported separately.