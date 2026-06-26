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
