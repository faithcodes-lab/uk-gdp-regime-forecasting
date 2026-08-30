# Why does XGBoost forecast the same value for 2026 Q1 and Q2?

Confirmatory diagnostic, not a model change.

**Same terminal leaf in every tree for the Q1 and Q2 rows despite different inputs: True.**

Across 91 non-crisis quarters, the model predicts 2 distinct values, the most common one 84 times (92%).

Prediction std: 0.0406. Actual std: 0.4860.

Constant-mean benchmark RMSE: 0.4885. Model RMSE: 0.4746 (2.9% improvement over always predicting the mean).
