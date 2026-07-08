# Decision Log

This file captures methodological and technical decisions made throughout the project. Each entry feeds into the reflection chapter of the dissertation and into viva preparation.

Format for each entry: the decision, the options considered, and the rationale.

---

## Decision: Repository and tooling setup
Date: 03-06-2026
Decision: Established the project repository structure, Python packaging, and continuous integration at the start of the project.
Rationale: A consistent structure and automated testing from day one supports reproducibility and reduces friction later in the project.

---

## Decision: Pipeline architecture: engineered pipeline vs Jupyter notebooks
Date: 07-06-2026
Question: How should the data + modelling workflow be structured?
Options considered:
  A. Jupyter notebook-first (a small number of large notebooks containing both pipeline and analysis)

  B. Engineered Python pipeline (src/ modules with type hints, YAML config, pandera schemas, lineage records, tests, Makefile, Docker)
Decision: B
Rationale:
  - Reproducibility is a Distinction-level marking criterion; an engineered pipeline allows `docker compose run pipeline make data` to reproduce the entire analysis on any machine, while a notebook depends on cell-execution order and the runtime environment.
  - The regime-shap package extraction in Sprint 5 is straightforward from engineered code, painful from notebook code.
  - The engineering rigour itself scores marks under the 15% Implementation criterion.
  - Cost: Sprints 1-2 are mostly engineering rather than analysis, but the trade is worth it given the SHAP analysis is computationally complex enough to need solid infrastructure.

---

## Decision: Data versioning:  no DVC
Date: 07-06-2026
Question: Use DVC (Data Version Control) to track dataset changes?
Options considered:
  A. Adopt DVC, track every raw download + processed dataset, push to remote storage

  B. Skip DVC; use the existing lineage records (data/lineage/*.json) and git-tracked schemas
Decision: B
Rationale:
  - The final dataset is small (104 quarters × 18 columns); the raw inputs are individually small and freely re-downloadable from public sources.
  - The lineage records already capture source, transformations, parameters, UTC timestamp, and git commit per artefact, which is the audit trail DVC would otherwise provide.
  - DVC adds tooling complexity and a remote-storage dependency without proportional benefit at this scale.

---

## Decision: PMI / business-confidence data source
Date: 07-06-2026
Question: How to source UK business confidence given S&P Global PMI has no free API?
Options considered:
  A. Scrape S&P Global (rejected, terms of service prohibit scraping)

  B. FRED proxy, OECD Business Tendency Surveys via FRED (BSCICP02GBM460S)

  C. Direct OECD SDMX-JSON

  D. Omit the series
Decision: B (FRED proxy)
Rationale:
  - The OECD/FRED series carries the same underlying signal (a percentage-balance survey indicator) and is already a recognised PMI proxy in the macro forecasting literature.
  - Using FRED keeps the dataset to two API styles (FRED + ONS website + BoE), avoiding a second OECD client. The OECD direct path was dropped from the pipeline entirely.
  - Verified the FRED series metadata: title "Business Tendency Surveys (Manufacturing): Confidence Indicators … for United Kingdom", units "Percent" (percentage balance), seasonally adjusted; same family applies to consumer_confidence via CSCICP02GBM460S.
  - Schema bound corrected to in_range(-60, 60) once the live data showed values in the percentage-balance range rather than the amplitude-adjusted index I initially assumed.

---

## Decision: COVID Q2 2020 outlier handling
Date: 09-06-2026
Question: How to handle the COVID Q2 2020 quarter, which printed the largest single-quarter GDP shock in the entire 2000-2025 sample (approximately -19.9% QoQ in the latest-vintage ONS data; first prints were closer to -19.4%)?
Options considered:
  A. Remove as outlier

  B. Winsorise or cap

  C. Keep as observed; widen schema bounds to accommodate; report bootstrap CIs and note small-sample caveat in Discussion
Decision: C
Rationale:
  - Removing the most informative shock in the sample would defeat the purpose of regime-aware SHAP analysis, the COVID regime is exactly the kind of structural break the dissertation studies.
  - Schema bounds set to ge=-25.0, le=25.0 on gdp_growth and gdp_lag_* columns accommodate the -19.9% observation with headroom.
  - The COVID regime contains only 6 quarters, so per-regime results are reported with bootstrap confidence intervals and an explicit small-sample caveat in the Discussion chapter.
  - A dedicated quality check (src/data/quality.py::_check_covid_outlier_present) asserts COVID Q2 2020 is present and below -10% on every pipeline run; if a future data revision changes that, the build fails loudly rather than silently producing a corrupted dataset.

---

## Decision: Yield-curve slope source: BoE GLC archive
Date: 13-06-2026
Question: Where to source 2-year and 10-year UK gilt yields for the yield-curve-slope feature?
Options considered:
  A. FRED has a clean 10-year (IRLTLT01GBM156N) but no 2-year UK sovereign series

  B. BoE Interactive Database (IADB) API currently used for Bank Rate and GBP/USD but does not expose constant-maturity gilt yields (term-structure data is in spreadsheets only)

  C. BoE Government Liability Curve (GLC) nominal monthly archive single ZIP file containing the full nominal spot curve at all maturities, monthly back to 1970
Decision: C
Rationale:
  - Both maturities must come from the same source for consistency; (A) and (B) each lack one leg.
  - The GLC archive is the official BoE-published constant-maturity series, methodologically the cleanest choice for a yield curve slope.
  - HTTP-200-HTML trap risk (BoE serves error pages with 200 OK and HTML body): downloader explicitly verifies the first four bytes are the ZIP signature PK\x03\x04 before parsing, so a bad response fails loudly instead of corrupting the dataset.
  - Both maturities live in the same "4. spot curve" sheet of every XLSX in the archive; an in-process cache lets the 2y and 10y downloads share a single network fetch.

---

## Decision: Drop Yahoo Finance / FTSE 100 from the dataset
Date: 09-06-2026
Question: Whether to include FTSE 100 quarterly return as a market predictor.
Options considered:
  A. Keep FTSE 100 via yfinance (the original spec)

  B. Drop it, keep the dataset focused on UK macro fundamentals
Decision: B
Rationale:
  - The dissertation framing is regime-aware SHAP analysis of UK GDP forecasting using macroeconomic predictors; equity-return signal is downstream of the macro factors already in the dataset (rates, inflation, confidence, oil), and including it adds variance more than information.
  - Removing yfinance simplified the dependency surface (one fewer third-party library known for reliability issues) and removed a source whose API terms change occasionally.
  - The final dataset has 17 data columns (1 target + 10 raw kept + 6 engineered), with yield_curve_slope active.

---

## Decision: gdp_yoy definition: 4-quarter compounded growth
Date: 12-06-2026
Question: How to define the "year-on-year GDP change" feature when the source series is a QoQ growth rate?
Options considered:
  A. Simple 4-quarter delta of the growth rate: g[t] − g[t-4]  (change in growth, momentum signal)

  B. Compounded 4-quarter growth: ((1+g[t])·(1+g[t-1])·(1+g[t-2])·(1+g[t-3]) − 1) × 100  (true level-on-level YoY)

  C. pct_change(periods=4) of the growth rate  (mathematically odd, denominator can be near zero)
Decision: B
Rationale:
  - Option B is the "true" YoY GDP growth, the level-on-level annual change that would be published in the press, reconstructed from QoQ rates. It is what an economist examining gdp_yoy would expect the feature to mean.
  - Computed in src/features/engineer.py::_cumulative_compounded_growth: input divided by 100 to convert percent to decimal, rolling product over 4 quarters, minus 1, back to percent. First 3 observations NaN (insufficient history; the first valid value is at the 4th observation), which is acceptable since the project window starts in 2000-Q1 and the raw ONS GDP series extends back to the 1950s.
  - cpi_yoy dropped at the same time because cpi_inflation is already a 12-month rate from ONS, so applying YoY transformation to it would be redundant.


---

## Decision: gfcf_growth and govt_consumption_growth derived from levels
Date: 15-06-2026
Question: How to source quarterly growth rates for Gross Fixed Capital Formation and Government Final Consumption, given the verified CDIDs NPQT and NMRY are levels rather than growth rates?
Options considered:
  A. Replace NPQT and NMRY with separate ONS growth-rate CDIDs (if they exist)

  B. Keep NPQT and NMRY (both CVM SA £m levels) and compute QoQ % change in the pipeline
Decision: B
Rationale:
  - Both series are chained-volume measures, seasonally adjusted (CVM SA): methodologically real, like-for-like quantities. A QoQ % change on these levels yields a real, seasonally adjusted growth rate on the same basis as the gdp_growth target (IHYQ, also CVM SA QoQ), so the three growth features are mutually consistent.
  - This mirrors how ONS itself derives the headline GDP growth print (level to QoQ %), avoiding any methodological asymmetry between gdp_growth and the two component-spending growth rates.
  - Implementation: new aggregation method `qoq_pct_change` in src/data/aggregate.py (resample to QE, take the period-over-period percent change × 100). First quarter is NaN by construction.
  - Surfaced during the first live `make data` run, where the raw £m levels failed the [-50, 50] growth-range schema check a useful catch.

---

## Decision: Feature engineering before date-window trim
Date: 15-06-2026
Question: Should features be engineered on the full available history and then trimmed to the 2000-2025 analysis window, or should the data be trimmed first and features engineered on the trimmed frame?
Options considered:
  A. Trim to 2000-2025 first, then engineer features (lags, rolling means, compounded gdp_yoy, yield-curve slope)

  B. Engineer all features on the full history first, then trim to 2000-2025
Decision: B
Rationale:
  - Engineering first means features at the 2000 boundary draw on genuine pre-2000 history rather than producing missing values; e.g. gdp_lag_4 at 2000-Q1 resolves to 0.7 rather than NaN, and the 4-quarter rolling mean and compounded gdp_yoy are fully populated from the first in-window quarter.
  - Trimming first would discard the history those backward-looking features depend on, forcing NaNs at the start of the window and silently shrinking the usable sample.
  - The raw ONS GDP series extends back to the 1950s, so there is ample history to feed the longest lookback (4 quarters) before 2000-Q1.

---

## Decision: Removed the deferred-feature scaffolding

Date: 14-06-2026
Question: Once yield_curve_slope was wired (2-year gilt available from the BoE GLC archive), is there any remaining need for the deferred-feature config flag, the skip branches in the downloaders, and the associated tests?
Options considered:
  A. Keep the scaffolding for hypothetical future deferral

  B. Delete every reference  the YAML deferred: true flags, the if entry.get("deferred"): continue branches in src/features/engineer.py and the downloaders, the "Skips deferred entries" docstring fragments, the count distinction between `final_column_count` and `final_column_count_when_slope_wired`, and the tests that asserted deferred behaviour
Decision: B
Rationale:
  - Nothing in the project is now deferred. The gilt-slope wiring closed the last open thread, and the deferred concept was bespoke scaffolding for that one situation.
  - Carrying dead code path forward made the column-count story confusing (two count fields where one would do) and the engineer-function logic harder to reason about.
  - Six tests that existed only to assert "deferred = no-op" behaviour were deleted; one duplicate count test was also dropped because it became equivalent to test_features_engineered_count_is_six.
  - The downloader return types tightened from pd.DataFrame | None to pd.DataFrame (None was only the deferred path).

---

## Decision: Number of regimes
Date: 23-06-2026
Question: How many regimes should the project use?
Options considered:
  A. 4 regimes (Pre-Brexit, Brexit, COVID, Post-COVID)

  B. 6 regimes (Pre-GFC Stability, Global Financial Crisis, Post-GFC Recovery, Brexit Transition, COVID-19 Shock, Post-COVID Recovery)

  C. 5 regimes (merging Post-GFC Recovery into Pre-GFC Stability)
Decision: B (6 regimes)
Rationale: Six regimes yield 15 pairwise SHAP comparisons (vs 6 for the 4-regime alternative), preserving statistical power for the novel-contribution analysis. Splitting the pre-Brexit window also keeps the GFC as a distinct test case rather than absorbing it into a homogeneous stable period. The trade-off is that the GFC (8 quarters) and COVID (6 quarters) regimes are small; this is handled by the bootstrap confidence intervals planned for the SHAP analysis. See the separate decision on small-sample regime handling.

---

## Decision: Statistical break detection method
Date: 23-06-2026
Question: Which method or methods should be used for statistical structural-break detection on UK quarterly GDP growth?
Options considered:
  A. Chow tests only (single known breakpoint at a time)

  B. Bai-Perron via the ruptures library only (multi-break, unknown breakpoints)

  C. Both Chow and Bai-Perron
Decision: C. Chow is applied at each of the five hypothesised boundaries; Bai-Perron is run as an unsupervised sensitivity sweep on the same series.
Rationale: The two methods answer different questions. Chow asks whether the regression differs on either side of a specific date, which is the natural test for literature-motivated boundaries. Bai-Perron asks where the data themselves would place breaks if no dates were imposed, providing independent statistical evidence. Running both makes agreement informative and disagreement transparent. The cost of running both is small (two short scripts; ruptures was already a planned dependency).

---

## Decision: Literature plus statistical dual justification for regime boundaries
Date: 23-06-2026
Question: How should each of the five regime boundaries be defended in the methodology chapter?
Options considered:
  A. Literature only (cite the GFC, Brexit, and COVID dates without formal tests)

  B. Statistical tests only (let Chow and Bai-Perron place every boundary)

  C. Both literature and statistical tests, with explicit convergence framing
Decision: C
Rationale: Literature alone is interpretable but offers no independent statistical evidence. Statistics alone is methodologically pure but loses the economic narrative (a break at 2008 Q3 is harder to defend than the onset of the global financial crisis). Combining the two converts methodological agreement into mutually reinforcing evidence and converts disagreement into a question the methodology chapter must address openly rather than hide. This was the supervisor's explicit recommendation in Meeting 1.

---

## Decision: Small-sample regime handling (GFC and COVID)
Date: 23-06-2026
Question: How should the two short regimes (GFC, 8 quarters; COVID, 6 quarters) be handled given the risk that per-regime statistics will be noisy?
Options considered:
  A. Merge GFC into Pre-GFC Stability and COVID into Post-COVID Recovery (drop the small regimes entirely)

  B. Exclude the small regimes from per-regime analysis but keep them in the full series

  C. Keep both regimes, flag the small sample sizes explicitly in every table and figure, and use bootstrap confidence intervals in the per-regime SHAP analysis
Decision: C
Rationale: Merging defeats the methodological point (the GFC is a structural shock of comparable importance to Brexit and COVID; pretending otherwise misrepresents the data-generating process). Excluding loses two key test cases for the novel-contribution analysis. Keeping the regimes with explicit small-sample caveats is methodologically honest and is the cited standard in regime-aware empirical work. Bootstrap confidence intervals for the per-regime SHAP statistics are implemented in 06-shap-analysis.md; this checkpoint commits to that approach.

---

## Decision: PELT penalty parameter (single value vs sensitivity sweep)
Date: 23-06-2026
Question: What penalty value should the Bai-Perron / PELT step use on the 104-quarter GDP series?
Options considered:
  A. A single penalty value chosen ex ante (commonly 10 in the macroeconomic literature)

  B. A sensitivity sweep across a grid of penalty values, reporting which breaks survive across the range
Decision: B (sensitivity sweep)
Rationale: A single penalty is arbitrary in this setting. Small-sample quarterly data is known to be sensitive to the penalty choice, and any single value invites the question "why that one". A sweep removes the degree of freedom from the analyst and lets the report state honestly which breaks are robust and which depend on the penalty. The implementation in src/regimes/run_analysis.py runs the default grid [5, 10, 15, 20, 30] and auto-widens to [1, 3, 5, 10, 15, 20, 30, 50, 100] when the default grid is uninformative (the same number of breaks at every penalty). The result is saved to results/regimes/bai_perron_sensitivity.csv.

---

## Decision: Add ICSS variance break test as a third instrument
Date: 23-06-2026
Question: How should the structural-breaks methodology respond to the finding that the mean-based Chow test (p = 0.952 at the 2008 boundary) and the rbf-PELT Bai-Perron sweep (no breaks at standard penalties 5 through 100; only a weak 2008-12-31 signal at penalty 1, the most sensitive setting, which tends to over-detect on a short series) did not flag the 2008 GFC at any defensible penalty, despite the GFC being one of the project's named regime boundaries?
Options considered:
  A. Keep Chow and Bai-Perron only and rely on literature alone to defend the GFC boundary

  B. Drop the GFC regime in light of the negative statistical evidence

  C. Add a variance-targeted change-point test as a third instrument, on the grounds that the GFC is a volatility event rather than a level event
Decision: C. The ICSS test (Inclan-Tiao 1994) is now a standard part of the structural-breaks methodology alongside Chow and Bai-Perron. It is implemented in src/regimes/volatility.py and orchestrated by src/regimes/run_analysis.py; results are written to results/regimes/icss_results.json with a GFC evidence block.
Rationale: Chow tests differences in regression coefficients (mean structure) and Bai-Perron with the rbf kernel is in principle sensitive to changes in distribution, but at standard penalties on a 104-quarter sample it returned no breaks. Neither instrument is calibrated to detect a pure variance shift, which is what the 2008 crisis was for UK GDP. ICSS targets the second moment directly via CUSUM-of-squares applied recursively. On the live series it detects two variance breaks inside the then-current 2008 Q1 to 2009 Q4 window (2008-06-30 with D = 2.993; 2009-09-30 with D = 3.319), providing evidence of a volatility shift inside the GFC regime as currently defined. Chow and Bai-Perron find evidence of mean shifts; ICSS finds evidence of volatility shifts. These are different kinds of structural change, and a test finding evidence of a shift does not by itself confirm a regime boundary.

Open item (separate from the test decision above): the exact GFC and Post-GFC Recovery boundary dates are still under discussion with the supervisor. Specifically, the GFC start date may move from 2008 Q1 (current config) to 2008 Q2 (closer to the first ICSS variance break at 2008-06-30), and the Post-GFC Recovery start date may move from 2010 Q1 (current config) to 2009 Q4 (closer to the second ICSS variance break at 2009-09-30). No boundary in config/regimes.yaml has been moved. Any change to the boundary dates will be recorded as a separate decision-log entry once the supervisor has confirmed. Resolved on 26-06-2026 by the Regime boundary dating rule entry.

---

## Decision: Regime boundary dating rule
Date: 26-06-2026
Question: How should the regime boundary dates be chosen in a consistent, defensible way, after the supervisor advised choosing boundaries by a clear rule rather than defending one set of dates as the single correct answer?
Options considered:
  A. Peak-based dating (a regime starts at the cyclical peak). Rejected, because it would force the COVID boundary back to 2019 Q4 to stay consistent, which is not sensible.

  B. Ad hoc dating per regime (choose each boundary individually). Rejected, because it is inconsistent and hard to defend in the viva.

  C. A single rule applied across all regimes: a regime begins in the first quarter that belongs substantively to the new state. For a crisis, the first quarter of GDP contraction; for a recovery, the first quarter of sustained renewed growth; for Brexit, the first full quarter after the referendum; the baseline starts at the data window. Chosen.
Decision: C. This moved two boundaries to comply with the rule:
  - GFC start from 2008 Q1 to 2008 Q2 (the first quarter GDP actually contracted)
  - Post-GFC Recovery start from 2010 Q1 to 2009 Q4 (the first quarter growth resumed)

  The other three internal boundaries (Brexit 2016 Q3, COVID 2020 Q1, Post-COVID 2021 Q3) and the 2000 Q1 baseline start already complied and were unchanged. New per-regime quarter counts: 33, 6, 27, 14, 6, 18, summing to 104.
Rationale:
  - The rule makes the GFC and COVID boundaries consistent with each other: both are now dated from the first contraction quarter. Under the old dates the GFC was peak-dated while COVID was contraction-dated, which was the inconsistency the rule removes.
  - Both moved dates have independent corroboration beyond the rule. 2008 Q2 is the technical recession start and is where the ICSS variance break falls. 2009 Q4 is when the UK officially exited recession and is close to the ICSS variance-down break at 2009 Q3.
  - Two judgement calls are acknowledged openly. First, the Pre-GFC Stability label now covers 2008 Q1; this is defensible because regimes are defined by GDP behaviour and GDP was still growing in 2008 Q1, even though the financial sector was already under stress. Second, the Post-COVID Recovery start at 2021 Q3 reflects sustained recovery; a brief rebound in 2020 Q3 was reversed by renewed restrictions, so the literal first quarter of renewed growth is not the right marker.
  - Alternative boundary dates remain plausible and this will be stated in the writeup. A contained sensitivity check on one alternative date set is planned if time allows, to show whether the headline results are robust to the choice.

---

## Decision: Business and consumer confidence series form
Date: 26-06-2026
Question: Whether to use the percentage-balance form of the OECD business and consumer confidence series (centred on zero) or the amplitude-adjusted form (centred on 100).
Options considered:
  A. Percentage-balance form (centred on zero), as currently stored in the dataset.

  B. Amplitude-adjusted form (centred on 100), which would require a config and pipeline change to swap the FRED series codes.
Decision: A, the percentage-balance form. A spot-check confirmed the stored series (business_confidence, consumer_confidence) are in percentage-balance form and match the FRED sources (BSCICP02GBM460S, CSCICP02GBM460S). No change needed before modelling.
Rationale: the percentage-balance form carries the same underlying survey signal and is a recognised confidence proxy. It is internally consistent with the rest of the feature set, and the spot-check found no data fault. The amplitude-adjusted form was considered but not needed; switching would add pipeline work for no analytical gain.

---

## Decision: Sprint 3 preprocessing and modelling protocol (pre-implementation)
Date: 27-06-2026
Question: What preprocessing, cross-validation, and model-comparison rules govern Sprint 3 modelling, agreed before any model code exists?
Options considered:
  A. Defer the rules and let them emerge during implementation.

  B. Record a single comprehensive protocol upfront, covering preprocessing discipline, per-model preprocessing, regime treatment, CV schemes, and model-comparison structure.

  C. Spread the rules across separate decision entries as each becomes relevant during implementation.

Decision: B. The eleven constraints below are adopted as the standing protocol for Sprint 3 modelling (04-modelling.md). They apply to ARIMA, Ridge, XGBoost, and LightGBM throughout the sprint and are not subject to change without a new decision-log entry.

Framing: Sprint 3 evaluates whether nonlinear machine learning models provide additional predictive value over
 (a) a univariate statistical baseline (ARIMA) and
  (b) a linear multivariate baseline (Ridge), so that any performance gain can be attributed separately to feature richness (ARIMA versus Ridge) and to nonlinearity (Ridge versus the gradient-boosting models). The comparison is deliberately three-way, not a two-way machine-learning-versus-statistics split, to avoid confounding those two effects. Any reference to explanatory value means value in explaining model behaviour via SHAP, which is exploratory and not a causal claim about the drivers of GDP.

Preprocessing:
  1. Leakage discipline is the top rule. Any step that learns from the data (scaling above all) must be fit inside each cross-validation fold on the training portion only, never on the whole series first. Fit on data through quarter t, apply to predict t+1, then expand and refit. Use a pipeline so the scaler is bound to the fold automatically.

  2. Scaling applies to Ridge only. Standardise (z-score) the features for Ridge. Do not scale for XGBoost or LightGBM; trees are scale-invariant. ARIMA ignores the feature matrix entirely.

  3. ARIMA preprocessing is a stationarity check on the target only. Run ADF or KPSS on gdp_growth and decide the differencing order d. The target is already a growth rate so it may be stationary at d=0; confirm rather than assume, and reuse the EDA stationarity result if available.

  4. ARIMA leakage discipline is separate from the feature pipeline. ARIMA forecasts from the target's own history, so at each one-step-ahead step refit ARIMA on gdp_growth through quarter t only, then predict t+1. Do not fit ARIMA once on the whole series.

  5. One-step-ahead alignment. Features at quarter t must map to the target at t+1. Confirm the feature row and target row are shifted so nothing contemporaneous with the predicted quarter leaks into its own prediction. Lag features are fine; verify the shift.

  6. The regime column is a grouping key for the regime-aligned CV folds, not a model input. It needs no encoding. Feeding regime in as a feature would be a separate logged decision, not a default.

  7. No NLP-style preprocessing (no stop words, tokenising, stemming); the data is numeric tabular time series. No missing-value imputation; the dataset has zero NaNs by the engineer-before-trim design. No outlier removal; the COVID quarter is real signal.

  8. random_state=42 everywhere a seed is needed.

Cross-validation and evaluation:
  9. Two CV schemes, kept distinct. Primary: expanding-window (walk-forward) one-step-ahead, the honest time-series scheme. Secondary: regime-aligned folds, so performance can be reported within each regime. They answer different questions and are both retained; do not collapse them into one.

  10. Store every prediction tagged with its quarter AND its regime from the first model onward, so per-regime metrics can be computed later. Per-regime reporting cannot be added retrospectively if predictions were not tagged as they were generated.

  11. The model comparison must be built to decompose two effects, per supervisor guidance: ARIMA versus Ridge isolates the value of adding features (univariate to multivariate, both linear); Ridge versus the gradient-boosting models (XGBoost, with LightGBM as a robustness check) isolates the value of non-linearity. Structure results so these two comparisons are visible, not just a four-way ranking by error.

Rationale: A pre-implementation protocol prevents leakage and per-regime-tagging defects that are hard to retrofit, locks in supervisor-recommended discipline (one-step-ahead, expanding-window CV, regime-aligned CV as a secondary scheme, and the two-comparison decomposition), and gives Sprint 4 evaluation a known shape to consume. Each constraint is independently defensible at viva and the bundled form makes the protocol easy to cite later.

---

## Decision: Minimum training size for the first expanding-window CV fold
Date: 30-06-2026
Question: What minimum training size should the expanding-window CV require for the first fold?
Options considered:
  A. No floor; the first training set is whatever the data size and the number of folds happen to leave over.

  B. A fixed minimum of 20 quarters (5 years), enforced in code.

Decision: B. The 20-quarter floor is enforced by the MIN_TRAIN_SIZE constant in src/models/cv.py; expanding_window_splits raises if len(X) is less than n_splits * test_size + MIN_TRAIN_SIZE.

Rationale: The first expanding-window fold needs enough history for the models to learn a meaningful relationship; 20 quarters (5 years of quarterly data) is a sensible floor that still leaves ample folds on the 104-quarter series. The full CV-scheme decision is deferred to CP7; this entry records only the floor.

---

## Decision: Cross-validation scheme
Date: 30-06-2026
Question: What CV scheme should Sprint 3 use for model performance estimation?
Options considered:
  A. Single expanding-window scheme.

  B. Single regime-aligned scheme.

  C. Both schemes, kept distinct, answering different questions (expanding-window primary, regime-aligned secondary).

Decision: C. Expanding-window CV is the primary scheme, used for hyperparameter tuning and primary performance reporting. Regime-aligned CV is the secondary scheme, used to assess generalisation to novel regime types. They are kept distinct and never collapsed into one number, per rule 9 of decision-log entry 20.

Rationale: Expanding-window is the standard time-series CV scheme and was supervisor-confirmed in Meeting 2. Regime-aligned is the methodological extension that directly tests how the model generalises to a regime it has not seen before, which is the question that motivates the SHAP regime-stability work in Sprint 4. The two schemes answer different questions and both are retained.

---

## Decision: Hyperparameter tuning approach
Date: 30-06-2026
Question: How should the three tunable models (Ridge, XGBoost, LightGBM) be tuned, given a 104-quarter dataset?
Options considered:
  A. Grid search over a small grid for each model.

  B. RandomizedSearchCV on the first 75 percent of data, with cached best-params JSON so downstream runs do not re-tune.

  C. Nested CV (inner CV for tuning, outer CV for performance estimation).

  D. Bayesian optimisation (Optuna or hyperopt).

Decision: B. RandomizedSearchCV on the first 75 percent of data (78 quarters), inner expanding-window CV with 5 folds, n_iter 20 for Ridge and 30 for XGBoost and LightGBM, random_state 42 throughout. Best-params results are cached to results/tuning/<model>_best_params.json so make train uses the cache by default and only retunes with --retune. ARIMA is tuned separately via select_arima_order (AIC grid search over p, d, q), not RandomizedSearchCV.

Rationale: RandomizedSearchCV is the practical compromise between full grid search (slow) and nested CV (computationally infeasible on a small dataset). Reserving the final 25 percent (26 quarters) for evaluation means the search never sees the held-out data, eliminating tuning-time leakage into evaluation. Caching keeps Sprint 4 evaluation cheap and reproducible across runs.

---

## Decision: Models included (three-way comparison)
Date: 30-06-2026
Question: Which forecasting models should be compared in Sprint 3, and how should the comparison be structured to avoid the machine-learning-versus-statistics confound?
Options considered:
  A. Two models: ARIMA versus XGBoost (univariate versus multivariate, but conflates feature richness with non-linearity).

  B. Three models: ARIMA, Ridge, XGBoost.

  C. Four models: ARIMA, Ridge, XGBoost, LightGBM, structured as a three-way comparison.

Decision: C. Four models, three-way comparison. ARIMA versus Ridge isolates the value of adding features (both linear, ARIMA univariate, Ridge multivariate). Ridge versus the gradient-boosting models (XGBoost and LightGBM) isolates the value of non-linearity (all multivariate, linear versus tree-based). LightGBM is the robustness check on the gradient-boosting result. Any performance gain can then be attributed separately to feature richness and to non-linearity rather than lumped together.

Rationale: The supervisor required this fairness framing because comparing ARIMA against XGBoost alone conflates two effects: adding features (univariate to multivariate) and adding non-linearity (linear to tree-based). Without the decomposition, a claim that XGBoost beats ARIMA would be ambiguous about whether the gain comes from features or from non-linearity. The three-way structure forces the distinction. This framing must carry into the writeup and the viva.

---

## Decision: Regime column treatment
Date: 30-06-2026
Question: Should the regime label be a model feature, a grouping key for the CV folds, or both?
Options considered:
  A. Regime as a model feature (one-hot encoded or treated as categorical).

  B. Regime as a grouping key only, used by regime_aligned_splits and for per-regime evaluation reporting; never passed to the models as a feature.

  C. Both: regime is a feature and a grouping key.

Decision: B. Regime is a grouping key only, not a model input. The regime column is dropped from the feature matrix before any model.fit call in train_all.py.

Rationale: Including regime as a feature would let the model trivially condition predictions on regime membership, which compromises the analysis of how feature importance shifts across regimes in Sprint 4. If a model can branch directly on regime, then the SHAP feature-importance ranks for the other features become harder to interpret as regime-specific signal. Holding regime out of the feature set keeps the per-regime SHAP comparison clean.

---

## Decision: Early stopping for gradient boosting
Date: 30-06-2026
Question: Should XGBoost and LightGBM use early stopping during training?
Options considered:
  A. Early stopping with a held-out validation split inside each CV fold.

  B. No early stopping; tune n_estimators explicitly via RandomizedSearchCV.

Decision: B. No early stopping. n_estimators is in the tuning grid (50, 100, 200, 500), so the search picks a sensible value. Final training uses the chosen n_estimators with no further stopping logic.

Rationale: Early stopping requires a held-out validation set inside each CV fold, which adds complexity to the per-fold leakage discipline (rule 1 of decision-log entry 20) and makes the comparison with Ridge and ARIMA less clean. Explicit n_estimators tuning is simpler, more interpretable, and matches the conservative grid agreed in CP3. The cost of not using early stopping on a 104-quarter dataset is small because the conservative max_depth (2 to 4) already limits overfitting.

---

## Decision: Forecast accuracy metrics for the dissertation
Date: 30-06-2026
Question: Which forecast accuracy metrics should constitute the headline evaluation suite, and what should the MASE denominator be?
Options considered:
  A. RMSE only.

  B. RMSE plus MAE plus R squared (no scale-invariant metric).

  C. RMSE, MAE, MASE, R squared, with MASE scaled per-fold by the local y_train (MASE values not comparable across folds, schemes, or regimes).

  D. RMSE, MAE, MASE, R squared, with MASE scaled by the full y series as a single common denominator.

Decision: D. Four metrics with MASE on a common full-series denominator.

Rationale: RMSE is the headline penalty (rule 5 of decision-log entry 20); MAE is the L1 companion. MASE adds the scale-invariant comparison against the no-skill naive baseline, which is what makes "below 1.0 beats the naive forecast" interpretable. R squared adds the explained-variance check. A common MASE denominator makes the value directly comparable across every row of the per-fold, per-regime, and aggregated tables; per-fold denominators would have produced six different MASE scales in the per-regime breakdown and broken the comparability.

---

## Decision: Diebold-Mariano significance test with HLN correction
Date: 30-06-2026
Question: How should pairwise model comparisons be tested for significance, and what corrections should be applied across the six-comparison family?
Options considered:
  A. Raw DM statistic with standard normal reference, no multiple-comparison correction.

  B. DM with HLN small-sample correction, t-distribution reference with n-1 degrees of freedom, no multiple-comparison correction.

  C. DM with HLN correction, t reference with n-1 df, Bonferroni correction across the six-comparison family with both raw and corrected p-values reported on the result.

Decision: C. HLN-corrected DM with Bonferroni and both raw and corrected p-values surfaced.

Rationale: The standard-normal reference is overconfident on the small (around 26 to 32 quarters) test sets the CV produces. Harvey, Leybourne and Newbold 1997 give the small-sample fix and recommend the t reference with n-1 df. With four models the comparison family has six pairwise tests, so Bonferroni is the conservative familywise-error control (rule 7 of decision-log entry 20). Reporting both raw and corrected p-values leaves the reader to judge how aggressive the correction should be; the dissertation will report the Bonferroni-corrected p-value as primary.

---

## Decision: Bootstrap confidence intervals for small-sample regimes
Date: 30-06-2026
Question: How should uncertainty be quantified for the GFC (n=6) and COVID (n=6) regimes where the standard asymptotic CIs are unsafe?
Options considered:
  A. Report no confidence intervals for small-sample regimes; let the point estimate stand alone.

  B. Block bootstrap respecting the autocorrelation structure of forecast errors.

  C. IID bootstrap with replacement (1000 iterations, 95 percent percentile CI, seed 42).

Decision: C. IID bootstrap with the limitation made explicit in the dissertation.

Rationale: Block bootstrap is the methodologically correct choice for autocorrelated errors but is infeasible at n=6: no usable block length exists between size 2 (too small to preserve correlation) and size 6 (no resampling possible). IID resampling is the only practical method. Acknowledged limitation: IID ignores any serial dependence in forecast errors and therefore understates uncertainty; the small-regime CIs are reported as indicative ranges, not precise intervals. The dissertation's limitations section names this trade-off directly so the reader can weigh it.

---

## Decision: Cross-validation aggregation across folds
Date: 30-06-2026
Question: How should per-fold metric values be aggregated into a single per-model, per-scheme summary?
Options considered:
  A. Pooled prediction-level metric across all folds (folds collapse into a single concatenated y_true and y_pred).

  B. Mean across folds weighted by fold size.

  C. Unweighted mean across folds with sample standard deviation (ddof=1) and median; NaN std on single-fold groups; per-fold observation counts preserved so the size-weighted view remains accessible to anyone who wants it.

Decision: C. Unweighted across-fold aggregation.

Rationale: Each fold is one independent train-test experiment; equal-fold weighting treats each experiment as one observation, which matches the experimental design. Size-weighting would privilege the late folds of the expanding-window scheme (they have the same test_size but the most surrounding context), conflating CV-fold structure with the metric value. Pooling folds discards the per-fold structure entirely and prevents the CV-variance figure (boxplot across folds) that demonstrates whether the cross-fold variation dominates the cross-model variation. Preserving n_observations on the per-fold DataFrame keeps the alternative open without changing the headline aggregation.

---

## Decision: Publication table and figure rendering
Date: 30-06-2026
Question: How should publication tables and figures for the dissertation be rendered, and which dependencies are acceptable?
Options considered:
  A. Pandas to_markdown plus pandas Styler.to_latex (requires both tabulate and jinja2 as runtime dependencies).

  B. Hand-built markdown plus pandas Styler.to_latex (requires jinja2 only).

  C. Hand-built markdown plus hand-built LaTeX with booktabs rules (no extra runtime dependency for table rendering).

Decision: C. Hand-built markdown and hand-built LaTeX, dependency-free.

Rationale: Originally scoped as option B; switched to C during CP6 implementation when Styler.to_latex raised AttributeError on missing jinja2. Hand-built LaTeX is around 25 lines, mirrors the hand-built markdown helper that the same principle already required, and keeps the rendering layer entirely dependency-free. Captions use hedged language ("appears to outperform", "results suggest"); definitive claims are reserved for comparisons where the Diebold-Mariano test supports them. Palette: viridis for the continuous heatmap, tab10 for the categorical bar and box plots; both are colour-blind accessible (Coblis pass deferred to dissertation-writing phase).

---

## Decision: Evaluation retrains per fold; persisted models reserved for SHAP
Date: 30-06-2026
Question: Should Sprint 4 evaluation reuse the joblib-persisted models from Sprint 3, or refit per fold inside the CV loop?
Options considered:
  A. Reuse the single fitted model from results/models/*.joblib for every CV fold (fast, but the model has already seen the held-out test data).

  B. Per-fold retraining for sklearn models and refit-per-step for ARIMA inside CP2's prediction generation; the persisted joblibs are not used in evaluation at all and are reserved for the downstream SHAP analysis in Sprint 5.

Decision: B. Per-fold retraining throughout; persisted joblibs reserved for Sprint 5 SHAP.

Rationale: Reusing the full-fit model for every CV fold leaks future information into the predictions on every fold except the last, because the persisted model was fit on the full 104 quarters including each fold's test indices. CV correctness requires the model state for fold k to depend only on data prior to fold k's test indices (rule 1 of decision-log entry 20). The persisted joblibs remain valuable for Sprint 5: SHAP analysis operates on the full-fit model so the feature-attribution story uses every available quarter. Sprint 4 and Sprint 5 therefore use the same model definitions but different model instances, and that separation is by design rather than by accident.

---

## Decision: Per-regime table interpretation caveats
Date: 01-07-2026
Question: How should the per-regime results be read, given the two structural properties of the CV schemes that the aggregation diagnosis surfaced (ARIMA numbers identical across schemes; n and regime coverage differing across schemes)?
Options considered:
  A. Present the per-regime numbers at face value and leave interpretation to the reader.

  B. Record three interpretation caveats bound to the per-regime tables so any reader, reviewer, or future author applies them consistently: n counts prediction instances not unique quarters under regime-aligned; sklearn regime-aligned per-regime metrics pool across multiple training snapshots; expanding-window never tests Global Financial Crisis or Post-GFC Recovery.

Decision: B. Record the three caveats as a durable interpretation layer over the per-regime tables.

Rationale: The post-hoc diagnosis of per-regime aggregation established the three caveats and they must bind how per-regime results are written up. First, under regime-aligned CV the fold structure tests each late regime once per fold from its first appearance forward, so n on that table counts prediction instances rather than unique quarters. For example, the COVID row shows n=24 in regime-aligned but the underlying data is 6 unique quarters tested in 4 folds; effective sample size for any bootstrap CI on such a subset is the unique-quarter count, not n, and CI width should be interpreted accordingly. Second, sklearn models produce a different prediction per fold because each fold trains on a different set of prior regimes, so the sklearn regime-aligned per-regime metric is a pooled error across several training snapshots rather than a single clean error for a single trained model; the number represents average error across training states. Third, the expanding-window scheme's training portion covers the entire pre-Brexit era, so Global Financial Crisis and Post-GFC Recovery never enter its test folds; their absence from the expanding-window per-regime table means "not tested," not "forecast well," and only the regime-aligned scheme can speak to those two regimes at all. These caveats do not change the code; they are the reading protocol the dissertation and any downstream analysis must apply.

---

## Decision: XGBoost's SHAP-eligible model relies on two features, not seventeen
Date: 04-07-2026
Question: The globally fit XGBoost model (Sprint 3 cached hyperparameters: learning rate 0.01, 50 trees, max depth 4) shows exactly zero SHAP importance on 15 of its 17 features in every one of the six regimes, with only gdp_growth (0.812) and gdp_lag_4 (0.188) ever nonzero. Is this a bug in the SHAP pipeline, under-fitting from overly conservative hyperparameters, or a genuine lack of exploitable signal in the macroeconomic predictors?
Options considered:
  A. Treat the zero-importance features as a SHAP computation bug and debug the pipeline.

  B. Treat it as under-fitting and re-tune XGBoost with more capacity (more trees, higher learning rate, deeper trees) before proceeding with the SHAP analysis.

  C. Investigate first without changing the committed model: cross-check SHAP importance against XGBoost's own built-in gain-based importance, and run a controlled capacity comparison to see whether more capacity helps or hurts cross-validated forecasting accuracy.

Decision: C. Investigated first; the result rules out both A and B.

Rationale: XGBoost's built-in gain-based feature_importances_ matches the SHAP-based importance exactly (gdp_growth 0.812, gdp_lag_4 0.188, all 15 other features exactly 0.000), so the zero importance is a real property of the fitted model, not a SHAP pipeline defect. A controlled capacity comparison then ruled out under-fitting: refitting with more trees (50 to 500, same low learning rate), a higher learning rate (0.01 to 0.1), and deeper trees (max depth 4 to 6) made the model use progressively more features (2, then 15, then 17, then 17 of 17), but expanding-window cross-validation RMSE worsened at every step (2.397, 2.632, 2.689, 2.703). If the macroeconomic features carried missing signal the baseline was failing to capture, added capacity should have improved held-out accuracy at least initially; instead it monotonically degraded, the signature of a model fitting noise in the extra features rather than recovering signal. The macroeconomic predictors therefore lack exploitable out-of-sample signal at this sample size (103 training rows, 17 features), and the model RandomizedSearchCV selected in Sprint 3 is, in substance, a two-feature autoregression on GDP's own recent history. This unifies with Sprint 4's evaluation finding that no model beat the naive mean and R-squared was negative overall: the near-null forecasting result and the near-total SHAP concentration on two autoregressive features are the same underlying finding seen from two different angles. It also explains why every pairwise Spearman rho across the six regimes comes out at 1.000: with only two features ever nonzero and the same two in every regime, there is essentially nothing for the ranking to reorder, so the appearance of stable explanations across regimes is a trivial consequence of the model's structure rather than evidence of the model reasoning consistently about a rich feature set under different economic conditions.

Open question for supervisor: how to frame the RQ4 novel contribution given that the stability result, as currently computed, is trivially true rather than a substantive finding about SHAP behaviour under regime change. Options to discuss include reframing the contribution around why a CV-selected model collapses to two features and what that implies for explainability in policy settings, treating the two-feature collapse itself as the headline finding, or examining whether a broader class of models shows the same pattern. No code or model changes have been made pending that discussion.

---

## Decision: Feature ablation confirms the two-feature finding independently
Date: 07-07-2026
Question: The SHAP zero-importance result and the capacity experiment both point to the fitted XGBoost relying only on GDP history. Does holding out whole groups of features confirm this a third way, and does the diagnostic setup match the main evaluation closely enough to be believed?
Options considered:
  A. Rely on the SHAP zero-importance result and the capacity experiment alone.

  B. Add a feature ablation as a third, independent line of evidence: refit XGBoost on the frozen data with the Sprint 3 cached hyperparameters under the same expanding-window CV, comparing the full 17-feature set against GDP-history-only (5 features) and macro-only (12 features).

Decision: B. The ablation confirms the finding and the setup is certified faithful.

Rationale: Under the same frozen dataset, the same expanding-window cross-validation, and the same cached hyperparameters as the main evaluation, GDP-history-only (gdp_growth, gdp_lag_1, gdp_lag_4, gdp_rolling_mean_4q, gdp_yoy) reaches RMSE 2.343 and MAE 1.835, which matches and slightly beats the full 17-feature model (RMSE 2.397, MAE 1.873): dropping all 12 macroeconomic features did not cost any accuracy, it marginally improved it. Macro-only (the 10 raw predictors plus yield_curve_slope and business_confidence_rolling_mean_4q) is clearly worse at RMSE 2.640, about 13 percent worse than GDP-history-only, so with no access to GDP's own recent values the model forecasts meaningfully less well. The full-feature baseline row reproduces the recorded evaluation RMSE exactly (2.3974 against the recorded 2.397391), which certifies the diagnostic uses the same data, cross-validation, and model as the headline results, so the ablation rows are directly comparable rather than a separate experiment. This is the third independent line of evidence alongside the SHAP zero-importance result and the capacity experiment, and all three agree: GDP history carries the forecasting signal, and the macroeconomic predictors add no exploitable out-of-sample information at this sample size (103 training rows). Results saved to results/feature-ablation.md. No model or pipeline change was made; this is a confirmatory diagnostic run with the existing hyperparameters.

---

## Decision: LASSO confirms no exploitable macro signal from the use-them direction
Date: 07-07-2026
Question: The SHAP, capacity, and ablation evidence all come from XGBoost ignoring the macro features. Does a linear feature-selection model, which actively seeks and keeps useful features, find any exploitable macro signal the tree model missed?
Options considered:
  A. Treat the two-feature finding as settled on the three existing lines of evidence, all from the ignore-them direction (XGBoost gives macro features zero importance).

  B. Add LASSO as a confirmatory model from the use-them direction: L1-penalised linear regression selects features by driving some coefficients to zero, so it will retain the macro features if they help. Run it under the same setup as Ridge (frozen data, one-step-ahead shift, StandardScaler pipeline, alpha tuned on the first 75 percent by expanding-window inner search, evaluated by the same 8-fold expanding-window CV with per-fold refit).

Decision: B. LASSO confirms the finding from the opposite direction.

Rationale: LASSO did not collapse to GDP history. Fit on the full 103 rows with its tuned alpha (0.0695), it kept 12 of 17 features with nonzero coefficients, including 7 macro features, and its single largest coefficient was a macro one (govt_consumption_growth at -0.75). So a linear feature-selector left to fit does assign substantial in-sample weight to the macro features. But that weight bought no out-of-sample accuracy: LASSO's expanding-window CV RMSE was 3.947 (MAE 3.329), the worst of every model tested except ARIMA, and notably worse than Ridge (2.756) despite the two differing only in penalty type. The macro signal LASSO latched onto is in-sample fit that does not generalise to held-out quarters. This confirms the two-feature finding from the use-them direction: XGBoost ignores the macro features and loses nothing, while LASSO uses them and forecasts worse, and both point to the macro features carrying in-sample fit but no exploitable out-of-sample signal at 103 training rows. Honest caveat recorded for the write-up: the GDP-history features are highly correlated with one another, and LASSO is known to be unstable in which member of a correlated group it retains (here it zeroed gdp_rolling_mean_4q while keeping gdp_yoy), so the exact selected and zeroed feature lists should not be over-interpreted feature by feature; the robust, reportable result is the poor out-of-sample RMSE, not the precise membership of the kept set. This is the fourth independent line of evidence for the finding, alongside SHAP zero-importance, the capacity experiment, and the feature ablation. Results saved to results/lasso-comparison.md. No model or pipeline change was made; LASSO is a confirmatory diagnostic, not a candidate for the main comparison.

---

## Decision: 2026 out-of-sample forecast, model choice, frozen training, and Q1 validation
Date: 08-07-2026
Question: How should the live 2026 out-of-sample forecast be produced and validated, which model should it use, and does the training window change to include 2026 data?
Decision: Refit XGBoost, the model selected in the Sprint 4 evaluation, on the frozen 2000 to 2025 history with the Sprint 3 cached hyperparameters, forecast one quarter ahead, and validate against the published ONS figure as each quarter is released. Training stays fixed at 2000 to 2025; the 2026 data is used only as predictor input, never as training data. Scope is Q1 and Q2 only; Q3 and Q4 are out of scope because those quarters have not happened yet, which is a matter of timing, not a limitation of the work.

Rationale: XGBoost was chosen because it was the best performer in the Sprint 4 evaluation: the lowest mean RMSE, MAE, and MASE across both the expanding-window and regime-aligned cross-validation schemes, and the only one of the four models with a positive R-squared anywhere (0.067 on the regime-aligned scheme). No retuning was done; the Sprint 3 cached hyperparameters (50 trees, max depth 4, learning rate 0.01) were reused so the forecast model is the same one the evaluation assessed. The training window did not change. The 104-quarter frozen dataset (103 training rows after the one-step-ahead shift) is unchanged, and the refit is saved as its own model file, results/models/xgboost_2026_q1_refit.joblib, separate from the reserved Sprint 3 model, so the two stay traceable. Each feature row is built from quarter t and predicts quarter t+1, so the Q1 2026 forecast comes from the 2025 Q4 feature row, the last row already in the frozen dataset, and needs no new 2026 data; the real 2026 Q1 predictor data instead feeds the Q2 2026 forecast under the same mapping. Vintage handling mattered in practice: ONS revised three quarters inside the frozen set since it was built (2024 Q1 from 0.8 to 0.7, 2024 Q4 from 0.3 to 0.4, 2025 Q1 from 0.7 to 0.6), and two of these fall inside the 2025 Q4 feature row (2024 Q4 via gdp_lag_4, 2025 Q1 via the rolling-mean and year-on-year window), so the GDP lag and rolling features were read from the frozen dataset rather than the freshly re-downloaded raw series, keeping the model's inputs consistent with its training vintage. On validation, the Q1 2026 forecast was 0.444 percent against the published ONS actual of 0.6 percent, an error of 0.156 percentage points. Read against XGBoost's typical error in the Post-COVID Recovery regime from the Sprint 4 evaluation (expanding-window MAE 0.42, RMSE 0.51), that error is about 0.37 times the regime MAE and 0.31 times its RMSE, well within the model's usual error band for that regime rather than a surprising miss. The ONS figure is the real, chained-volume, seasonally adjusted, quarter-on-quarter growth measure (series IHYQ), the same target the model was trained on. The Q2 2026 forecast is still to be generated from the 2026 Q1 feature row (its predictor data is now complete) and will be validated once ONS publishes the Q2 figure, expected August 2026. Results saved to results/2026-q1-forecast.md and results/2026-q1-validation.md, with the predictor coverage report at results/2026-predictor-coverage.md.
