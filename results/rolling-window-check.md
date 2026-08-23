# Rolling-window check: is the near-null finding a side effect of expanding-window CV?

Confirmatory diagnostic, not a model change. Reconstructed methodology (see script docstring): no committed script, results file, or decision-log entry previously existed for this test.

Fixed training window of 71 quarters, sliding forward, with test fold boundaries identical to the primary expanding-window scheme (8 folds, 4 quarters each) so the two are directly comparable; same cached XGBoost hyperparameters as the main evaluation.

## Accuracy comparison

| Scheme | RMSE | MAE |
|---|---|---|
| Expanding-window | 2.397 | 1.873 |
| Rolling-window | 2.392 | 1.868 |

RMSE difference: -0.006 (-0.2%).

The two schemes give a similar result, so the near-null finding does not appear to be a side effect of the training set growing under the expanding-window scheme.
