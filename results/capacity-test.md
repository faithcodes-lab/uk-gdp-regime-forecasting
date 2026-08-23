# Capacity test: is XGBoost ignoring the macro features because it is underfitting?

Confirmatory diagnostic, not a model change. If the cached hyperparameters were too constrained to pick up a real macro signal, giving the model more capacity should let it find that signal and improve held-out accuracy. Each variant adds capacity on top of the previous one (a cumulative build-up, not isolated single-parameter toggles from the baseline); active features is a single fit on the full history with nonzero gain-based feature_importances_, RMSE/MAE are the mean across the same 8-fold expanding-window CV as the main evaluation.

## Effect of added model capacity

| Capacity change | Active features | Expanding-window RMSE | MAE |
|---|---|---|---|
| Baseline (50 trees, depth 4, lr 0.01) | 2 | 2.397 | 1.873 |
| More trees (50 to 500) | 15 | 2.632 | 2.036 |
| Higher learning rate (0.01 to 0.1) | 17 | 2.689 | 2.094 |
| Deeper trees (depth 4 to 6) | 17 | 2.703 | 2.093 |

Accuracy worsened at every step away from the baseline: yes. Active features rose from 2 in the baseline to 17 in the most complex variant, but cross-validated error got worse rather than better, the signature of a model fitting noise in the extra features rather than recovering signal. This rules out underfitting as the reason the baseline model ignores the macroeconomic predictors.
