**Overall forecast accuracy (regime-aligned CV). Lower values indicate better performance for RMSE, MAE, and MASE; higher R2 indicates better fit. Mean across folds with standard deviation in parentheses.**

| Model | RMSE | MAE | MASE | R2 |
| --- | --- | --- | --- | --- |
| arima | 7.750 (4.249) | 2.603 (1.638) | 2.265 (1.425) | -4.031 (0.058) |
| lightgbm | 3.526 (1.744) | 1.360 (0.623) | 1.183 (0.542) | -0.376 (0.832) |
| ridge | 4.056 (1.472) | 1.620 (0.575) | 1.410 (0.500) | -3.763 (8.080) |
| xgboost | 3.446 (1.933) | 1.314 (0.739) | 1.143 (0.643) | 0.067 (0.177) |