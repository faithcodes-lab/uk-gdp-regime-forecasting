# Completion Log: 04-modelling.md

## Session metadata

Covers prompts/04-modelling.md on branch feature/sprint-3-modelling, across sessions from 2026-06-27 to 2026-06-30.

## Tasks completed

### Task 1: cross-validation framework

Files created: src/models/cv.py with expanding_window_splits and regime_aligned_splits, src/models/visualise_cv.py with plot_cv_splits, tests/test_cv.py. The two splitters return list[(train_idx, test_idx)] tuples. expanding_window_splits is the primary scheme (8 folds of test_size 4 by default) with a guard requiring at least MIN_TRAIN_SIZE 20 quarters for the first fold. regime_aligned_splits derives fold structure from the regime column, producing r minus 1 folds for r regimes (5 folds for the project's 6 regimes). The three named leakage tests (no train test overlap, max train idx strictly less than min test idx, regime-aligned trains on first k regimes only) prove no fold sees future data. Verification: 13 tests pass at 100 percent coverage on both cv.py and visualise_cv.py.

### Task 2: forecasting interface and Ridge baseline

Files created: src/models/interface.py with the ForecastingModel abstract base class (fit, predict, get_params), src/models/ridge.py with build_ridge_pipeline and RidgeForecastingModel, tests/test_ridge.py. Ridge is wrapped in a sklearn Pipeline with StandardScaler so the scaler is fit per fold automatically. RidgeForecastingModel is a thin custom wrapper around the Pipeline. The scaling-discipline test (the leakage heart of Task 2) fits the pipeline on a slice and asserts scaler.mean_ matches the slice mean and differs from the full-X mean, proving no leakage from the held-out portion into the scaler's parameters. Verification: 11 tests pass at 100 percent coverage on both interface.py and ridge.py.

### Task 3: XGBoost and LightGBM gradient boosting

Files created: src/models/xgboost_model.py with build_xgboost_pipeline and XGBForecastingModel, src/models/lightgbm_model.py with build_lightgbm_pipeline and LGBMForecastingModel, tests/test_xgboost.py, tests/test_lightgbm.py. Both wrappers expose the same ForecastingModel interface as Ridge. Each Pipeline is single-step with no scaler since trees are scale-invariant; the no-scaler test for each model proves this. LightGBM pins min_child_samples to 5, overriding the library default of 20 which is too high for a 104-row dataset. Conservative defaults across both models: max_depth 3, learning_rate 0.1, n_estimators 200. Verification: 19 tests pass at 100 percent coverage on both modules.

### Task 4: ARIMA univariate baseline

Files created: src/models/arima.py with ARIMAModel, the select_arima_order helper, and _fit_with_fallback. Extended src/models/cv.py with cross_validate_arima. tests/test_arima.py plus three new tests in tests/test_cv.py. ARIMAModel conforms to ForecastingModel: fit takes (X, y) but ignores X; predict returns a static multi-step forecast from the fit-time state; predict_one_step_ahead refits on a history slice as the unit operation for refit-per-step CV. select_arima_order is an AIC grid search substituting for pmdarima.auto_arima, which is unavailable on Python 3.13. The leakage proof for cross_validate_arima asserts that predictions are unchanged when y values past the last test position are removed; any future leak would break this. The AR(1) coefficient recovery test fits ARIMA(1,0,0) on 500 synthetic AR(1) points and confirms the recovered phi is within 0.05 of the true value. Verification: 12 ARIMA tests plus 3 cross-validate-arima tests pass; coverage 93 percent on arima.py and 96 percent on cv.py at this task.

### Task 5: hyperparameter tuning

Files created: src/models/tune.py with tune_ridge, tune_xgboost, tune_lightgbm, plus save_tuning_result and load_tuning_result cache helpers, tests/test_tune.py. Each tune function runs RandomizedSearchCV with expanding-window CV on the first 75 percent of the data (78 quarters out of 104). The remaining 25 percent stays untouched for Sprint 4 evaluation. Ridge uses 20 alpha values from logspace minus 3 to 2 sampled at n_iter 20. XGBoost and LightGBM each sample 30 random combinations from their conservative grids. Scoring is neg_root_mean_squared_error so the best_score is interpretable as negated RMSE in target units. Best params are cached to results/tuning/<model>_best_params.json so downstream runs use the cache by default. ARIMA is not tuned here; its order comes from Task 4's select_arima_order. The methodological heart test captures the X handed to the splitter via monkeypatch and asserts it equals X.iloc[:int(0.75 * n)] in both length and content, proving the search never touches the held-out 25 percent. Verification: 9 tests pass at 92 percent coverage on tune.py.

### Task 6: master training script

Files created: src/models/train_all.py with main, prepare-data helpers, per-model train functions, and a persist helper. Extended Makefile with tune and train targets. Added tests/test_train_integration.py. The CLI is python -m src.models.train_all [--retune]. Default behaviour loads cached tuning results from results/tuning/ and falls back to running the tuner for any cache miss. The --retune flag forces re-tuning of all four models. Data prep is split by model: sklearn models train on features with the target shifted one quarter ahead (X.iloc[:-1] and y.iloc[1:].dropna), so X at quarter t predicts y at t+1; ARIMA trains on the unshifted gdp_growth series since it handles its own autoregressive lag internally. Each model is persisted as results/models/<name>.joblib with a sibling _meta.json carrying library versions, training timestamp, dataset md5, training row count, and random_state. The alignment proof asserts y at row i equals gdp_growth at row i+1 with the last row dropped; the reproducibility test runs main twice on synthetic data and asserts byte-identical .joblib files for all four models. Verification: 9 tests pass at 96 percent coverage on train_all.py.

### Task 7: Sprint 3 decisions and CV visualisation

Files modified: report/decision-log.md (five entries appended, 22 through 26), src/models/visualise_cv.py (added main and _save helper), Makefile (added figure-cv-splits target), tests/test_cv.py (added one smoke test for main). The five decision entries record the bundled modelling decisions deferred from earlier tasks: the two CV schemes kept distinct, the tuning approach, the four-model three-way comparison framing, regime as a grouping key not a feature, and no early stopping for the gradient-boosting models. The visualisation script writes both schemes as PNG at 300 dpi and PDF (vector) under results/figures/. The smoke test monkeypatches the data path and output dir to tmp_path, runs main on synthetic data, and asserts all four files exist. Verification: 17 cv tests pass at 98 percent coverage on visualise_cv.py.

## ARIMA integrity bug

A real correctness bug surfaced during the first end-to-end run of make train, not during the test suite. The Task 4 implementation of _fit_with_fallback called statsmodels' ARIMA.fit with method="lbfgs" and on failure with method="powell". In statsmodels 0.14, the method kwarg selects the estimation technique (statespace, innovations_mle, hannan_rissanen, and so on), not the optimiser. Both kwargs raised "not a valid estimator" exceptions, the third-tier fallback always fired, and the trained ARIMA was always ARIMA(1, 0, 0) regardless of what select_arima_order had chosen. The metadata JSON recorded the chosen order (for example (3, 0, 0)) while the persisted model was a different order; this is a correctness and audit-trail issue.

The Task 4 test suite missed this because the fallback test mocked _StatsmodelsARIMA to raise unconditionally on any bad order, so the mock never exercised the real .fit signature. The bug only surfaces against the real statsmodels API.

The fix in commit 2ee46bf drops method= entirely on the first attempt (the default optimiser is lbfgs internally) and on retry passes method_kwargs={"method": "powell"}. A new test, test_arima_fit_actually_uses_requested_order_not_fallback, fits ARIMA(3,0,0) on real AR(1) data and asserts the used_order returned by _fit_with_fallback matches (3,0,0). This is a real-statsmodels test, no mocks, so any future API misuse will fail immediately. After the fix, arima_meta.json correctly records order [3,0,0] and no fallback warning fires.

## Reproducibility finding

Ridge, XGBoost, and LightGBM joblib files are byte-identical across consecutive make train runs with the same seed and data; the integration test test_train_all_joblib_bytes_are_reproducible asserts this for all four models on synthetic data and passes. ARIMA on the real 104-quarter dataset is not byte-identical across runs: statsmodels embeds nondeterministic internal state in its pickled ARIMAResults (likely optimiser history or timestamps), so the bytes differ even though the fitted order and coefficient values are identical and the ARIMA fit is itself deterministic. The takeaway: three of four models byte-reproducible; ARIMA reproducible in fitted form (order and parameters) but not in pickled bytes. The amended commit message on 2ee46bf is explicit about this so the limitation is documented at the audit point.

## Final verification

225 tests pass across the whole repository in 23.82 seconds, covering Sprint 2 data pipeline, Sprint 2 structural breaks, and all seven Sprint 3 tasks. ruff check on src/ and tests/ is clean. black --check on src/ and tests/ is clean (56 files unchanged).

Sprint 3 per-module coverage: cv.py 96 percent, arima.py 95 percent, tune.py 92 percent, train_all.py 96 percent, visualise_cv.py 98 percent; ridge.py, interface.py, xgboost_model.py, and lightgbm_model.py all at 100 percent.

Aggregate coverage across the whole src tree is 79 percent. This is dragged down by two runtime-only orchestrator scripts that pytest does not exercise: src/data/build_dataset.py (87 statements, 0 percent), invoked only via make data; and src/regimes/run_analysis.py (137 statements, 0 percent), invoked only via make break-tests. Both have been exercised end-to-end in their respective sprints. Coverage on Sprint 3 modelling code is well above the 80 percent project gate.

make tune and make train were run end-to-end on the real 104-quarter dataset. All four trained models are committed to results/models/ alongside their _meta.json files as fixed reference inputs for Sprint 4 evaluation and SHAP analysis. The tuning caches at results/tuning/ are committed too so the cache-hit path on make train is reproducible by anyone with the repository. make figure-cv-splits produced the two CV visualisation PNGs (300 dpi) and PDFs (vector) under results/figures/, also committed.

## Decisions added to report/decision-log.md

Sprint 3 added 7 entries (20 through 26). On the Sprint 3 branch the decision log now holds 26 entries total.

20. Sprint 3 preprocessing and modelling protocol (pre-implementation): the bundled eleven-rule protocol covering leakage discipline, per-model preprocessing, regime treatment, CV schemes, and the three-way comparison structure, agreed before any model code was written.

21. Minimum training size for the first expanding-window CV fold: 20 quarters (5 years) enforced by MIN_TRAIN_SIZE in cv.py.

22. Cross-validation scheme: expanding-window primary plus regime-aligned secondary, kept distinct.

23. Hyperparameter tuning approach: RandomizedSearchCV on the first 75 percent of data with cached best-params JSON; ARIMA handled separately via select_arima_order.

24. Models included (three-way comparison): ARIMA, Ridge, XGBoost, LightGBM, with the supervisor-mandated decomposition where ARIMA versus Ridge isolates feature richness and Ridge versus the gradient-boosting models isolates non-linearity.

25. Regime column treatment: grouping key only, never a model feature.

26. Early stopping for gradient boosting: not used; n_estimators tuned explicitly instead.
