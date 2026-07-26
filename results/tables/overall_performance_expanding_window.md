**Overall forecast accuracy (expanding-window CV). Lower values indicate better performance for RMSE, MAE, and MASE; higher R2 indicates better fit. Mean across folds with standard deviation in parentheses.**

| Model | RMSE | MAE | MASE | R2 |
| --- | --- | --- | --- | --- |
| arima | 4.835 (10.452) | 3.849 (7.982) | 3.350 (6.947) | -5.657 (8.056) |
| lightgbm | 2.497 (4.470) | 1.961 (3.467) | 1.707 (3.017) | -6.161 (8.644) |
| ridge | 2.756 (4.938) | 2.225 (3.750) | 1.937 (3.264) | -6.012 (10.476) |
| xgboost | 2.397 (4.526) | 1.873 (3.491) | 1.630 (3.038) | -3.457 (6.121) |