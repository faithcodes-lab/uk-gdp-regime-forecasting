# Completion Log: 05-evaluation.md

## Session metadata

Covers Sprint 4 evaluation on branch feature/sprint-4-evaluation, completed on 2026-06-30. Seven checkpoints (CP1 through CP7) delivered in sequence: core metrics, prediction generation, Diebold-Mariano significance test, per-regime evaluation, results aggregation, publication tables and figures, and master orchestration plus decision entries.

## Tasks completed

### CP1: Core forecasting metrics module

Files created: src/evaluation/__init__.py, src/evaluation/metrics.py with compute_rmse, compute_mae, compute_mase, compute_r2, and compute_all_metrics, tests/test_metrics.py. RMSE wraps sklearn.metrics.root_mean_squared_error (sklearn 1.4+). MAE wraps mean_absolute_error. R squared wraps r2_score. MASE is a custom implementation following Hyndman and Koehler 2006 with the non-seasonal naive denominator computed on a provided y_train series; this lets the caller use the full target series as a common denominator so MASE is comparable across folds, schemes, and regimes in the same evaluation table. compute_all_metrics returns a dict mapping the four metric names to their values. Verification: 19 tests pass at 100 percent coverage on metrics.py.

### CP2: Prediction generation pipeline

Files created: src/evaluation/predictions.py with generate_predictions, _predict_sklearn_per_scheme, _predict_arima_per_scheme, and _sklearn_kwargs_from_best_params, tests/test_predictions.py. generate_predictions consumes the processed dataset, builds the common 103-row target space (target_df = df.iloc[1:]), runs all four models across both CV schemes, and returns a 992-row long-format DataFrame with columns model, quarter, regime, y_true, y_pred, fold_idx, scheme. sklearn models retrain per fold; ARIMA refits per step via cross_validate_arima with a +1 index offset that aligns ARIMA's natural (history at t predicts t+1) shape to the common target_df indexing. Best parameters are loaded from results/tuning/ caches. Verification: 10 tests pass at 98 percent coverage on predictions.py.

### CP3: Diebold-Mariano significance test

Files created: src/evaluation/diebold_mariano.py with the DMTestResult dataclass and diebold_mariano_test function, tests/test_diebold_mariano.py. Implements the Diebold and Mariano 1995 statistic with the Harvey, Leybourne and Newbold 1997 small-sample correction and the recommended t reference with n minus 1 degrees of freedom. Special case: identical forecasts return p=1.0 to avoid the 0/0 in the DM ratio. n_comparisons parameter sets the Bonferroni denominator visible on the returned dataclass, so a family of pairwise tests carries both raw and corrected p-values explicitly. Verification: 13 tests pass at 94 percent coverage on diebold_mariano.py.

### CP4: Per-regime evaluation framework

Files created: src/evaluation/regime_evaluation.py with evaluate_per_regime and bootstrap_regime_metrics, tests/test_regime_evaluation.py. evaluate_per_regime groups CP2's predictions by (model, scheme, regime) and reports the four metrics plus a small_sample boolean flag (default threshold: n < 10 quarters). bootstrap_regime_metrics computes 95 percent percentile-based bootstrap confidence intervals via IID resampling (1000 iterations, seed 42); intended to be called on the flagged small-sample regimes only. Degenerate resamples (constant y_true after resampling) are silently skipped. Verification: 12 tests pass at 90 percent coverage on regime_evaluation.py.

### CP5: Results aggregation across CV folds

Files created: src/evaluation/aggregation.py with compute_per_fold_metrics and aggregate_cv_results, tests/test_aggregation.py. compute_per_fold_metrics groups CP2's predictions by (model, scheme, fold_idx) and reports the four metrics plus n_observations. aggregate_cv_results then groups the per-fold metrics by (model, scheme) and reports mean, median, and sample standard deviation (ddof=1) per metric; single-fold groups yield NaN std. Aggregation is unweighted across folds; the per-fold n_observations is preserved so anyone can compute size-weighted statistics or the CV-variance boxplot directly. Verification: 10 tests pass at 93 percent coverage on aggregation.py.

### CP6: Publication tables and figures

Files created: src/reporting/__init__.py, src/reporting/tables.py with make_overall_performance_table, make_per_regime_table, and make_dm_test_table, src/reporting/figures.py with plot_model_comparison_bar, plot_regime_performance_heatmap, and plot_cv_fold_variance, tests/test_reporting.py, tests/conftest.py. Each table function returns a dict with markdown and latex keys; both formats are built by hand (no tabulate or jinja2 dependency). Originally scoped to use pandas Styler.to_latex but switched to hand-built LaTeX with booktabs rules during implementation when Styler raised AttributeError on the missing jinja2 optional dependency. Captions use hedged language ("appears to outperform", "results suggest"); definitive claims are reserved for comparisons where the Diebold-Mariano test supports them. Each figure function returns a matplotlib Figure; the orchestrator handles saving. Palette is viridis for the continuous heatmap and tab10 for the categorical bar and box plots. conftest.py sets the matplotlib Agg backend at session start so new test modules can import pyplot at the top without per-file noqa comments. Verification: 12 tests pass at 99 percent coverage across src/reporting/.

### CP7: Master evaluation script and decisions

Files created: src/evaluation/orchestrator.py with run_evaluation and four private helpers, scripts/run_evaluation.py with the argparse CLI wrapper, tests/test_orchestrator.py. Makefile extended with the evaluate target. report/decision-log.md extended with six Sprint 4 decision entries (the file now holds 32 entries total). run_evaluation reads the processed parquet, calls generate_predictions, runs the CP3 to CP6 stages in sequence, and writes the full artifact tree to results/{predictions,metrics,tables,figures}/. Tests mock generate_predictions to short-circuit the expensive per-fold retraining (which has its own coverage in tests/test_predictions.py) and exercise the orchestrator's wiring, artifact writing, and determinism on synthetic predictions. The determinism test is the methodological heart: it runs run_evaluation twice with the same seed on the same synthetic data and asserts that per_fold.csv, aggregated.csv, per_regime.csv, and dm_test.csv are byte-identical between runs, proving reproducibility of every persisted evaluation artifact. Verification: 5 tests pass at 86 percent coverage on orchestrator.py.

## Real-data end-to-end run

make evaluate ran end-to-end on the 104-quarter dataset in 11.7 seconds and produced 992 prediction rows. Artifact tree written:

- 1 file under results/predictions/ (predictions.parquet)
- 5 files under results/metrics/ (aggregated.csv, per_fold.csv, per_regime.csv, small_sample_cis.csv, dm_test.csv)
- 12 files under results/tables/ (three tables, two schemes, two formats each)
- 12 files under results/figures/ (three figure types, two schemes, PNG plus PDF each; RMSE only by default)

Headline findings from the real run, on the expanding-window CV scheme:

- Mean RMSE: XGBoost 2.40, LightGBM 2.50, Ridge 2.76, ARIMA 4.84. Std across folds is wide (4.5 to 10.5), indicating most of the cross-model spread is within the per-fold variance band.
- Mean MASE all between 1.6 and 3.4, none below 1.0 against the naive baseline.
- R squared negative for all four models, meaning none beats predicting the mean on a fold-averaged basis.
- All six Diebold-Mariano pairwise comparisons return Bonferroni-corrected p = 1.000; no significant difference in forecast accuracy between any model pair at the 5 percent level on either CV scheme.

The honest takeaway, to be expanded in the dissertation: UK GDP at quarterly frequency over 104 observations is genuinely hard to forecast with the chosen feature set, and there is no statistical evidence within this sample that any of the four models outperforms the others. This is the result the evaluation framework is designed to report cleanly, including the negative findings.

The COVID-19 Shock regime (n=6) inflates the per-regime RMSE substantially: ARIMA records RMSE 25.1, the gradient boosters around 11, ridge 12. Bootstrap CIs for this regime are correspondingly wide (lower bound around 3, upper bound around 16 for the gradient boosters). The small-sample asterisk and the IID-bootstrap limitation are both flagged in the rendered tables and the per-regime CSV.

## Final verification

306 tests pass across the whole repository (Sprint 4 contributed 81 tests: 19 CP1, 10 CP2, 13 CP3, 12 CP4, 10 CP5, 12 CP6, 5 CP7). Whole-repo aggregate coverage is 83 percent, above the 80 percent project gate. Sprint 4 module coverage averages 94 percent across src/evaluation/ and src/reporting/. ruff check on src/, tests/, and scripts/ is clean. black --check is clean on every file CP1 through CP7 touched (two pre-existing scripts/ files outside the Sprint 4 scope are flagged by black; left alone as they predate the formatting convention and would belong in a separate cleanup).

Per-module Sprint 4 coverage: metrics.py 100 percent, predictions.py 98 percent, diebold_mariano.py 94 percent, regime_evaluation.py 90 percent, aggregation.py 93 percent, orchestrator.py 86 percent, src/reporting/tables.py 100 percent, src/reporting/figures.py 98 percent.

Post-hoc diagnosis of the per-regime aggregation confirmed correct behaviour, not a bug. ARIMA's per-regime numbers are identical to the decimal across the two CV schemes because ARIMA refits per step on each quarter's own history and ignores the training set, so mean-based metrics computed on six unique COVID predictions and on the same six predictions duplicated four times (as happens under regime-aligned) are algebraically equal. The differing n values and regime coverage across schemes (COVID n=6 expanding-window against n=24 regime-aligned; Global Financial Crisis and Post-GFC Recovery absent from the expanding-window table entirely) reflect the two schemes' different test structures: expanding-window tests each late quarter once, regime-aligned tests each late regime once per fold from its first appearance forward, and expanding-window's training portion covers the pre-Brexit era so those earlier regimes never enter its test folds. The three interpretation caveats this diagnosis implies are recorded as a new decision-log entry.

## Decisions added to report/decision-log.md

Sprint 4 added six entries (27 through 32). On the Sprint 4 branch the decision log now holds 32 entries total.

27. Forecast accuracy metrics for the dissertation: RMSE, MAE, MASE, R squared with MASE on a single common denominator (the full y series), so MASE values are comparable across every row of the per-fold, per-regime, and aggregated tables.

28. Diebold-Mariano significance test with HLN correction: HLN small-sample correction, t reference with n minus 1 df, Bonferroni for the six-comparison family with both raw and corrected p-values surfaced on the result.

29. Bootstrap confidence intervals for small-sample regimes: IID bootstrap (1000 iters, 95 percent CI, seed 42) for the GFC (n=6) and COVID (n=6) regimes. Block bootstrap infeasible at n=6 and the IID limitation (ignores error autocorrelation, understates uncertainty) is recorded in the rationale and will appear in the dissertation's limitations section.

30. Cross-validation aggregation across folds: unweighted across folds with ddof=1 sample std and NaN std on single-fold groups; n_observations preserved per-fold so the size-weighted view remains accessible.

31. Publication table and figure rendering: hand-built markdown and hand-built LaTeX with booktabs rules, no tabulate or jinja2 dependency. Viridis (continuous), tab10 (categorical). Hedged caption language; definitive claims only where Diebold-Mariano supports them.

32. Evaluation retrains per fold; persisted models reserved for SHAP: sklearn retrain per fold and ARIMA refits per step inside CP2's prediction generation. Sprint 3's results/models/*.joblib files are not used in evaluation and are reserved for the Sprint 5 SHAP analysis. The same model definitions, different model instances, for the two sprints.
