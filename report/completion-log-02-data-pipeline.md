# Completion Log: 02-data-pipeline.md

## Session metadata

Covers prompts/02-data-pipeline.md on branch feature/sprint-2-data-pipeline, across sessions from 2026-06-07 to 2026-06-16.

## Tasks completed



### Task 1: configuration foundation

Files created: config/pipeline.yaml, config/features.yaml, config/regimes.yaml, src/data/config.py, tests/test_config.py. Declared 13 raw series (later evolved, see TASK 3 for the Yahoo drop and the source-set churn). Six regimes from 00-project-context.md section 4 captured with native YAML dates, quarter counts sum to 104. Loaders are functools.lru_cache-wrapped and return parsed dicts. Verification: 16 out of 16 config tests passed at the time of task 1; cascade trimmed over task 4 to task 5, see Final Verification for the current count.

### Task 2: foundation infrastructure

Files created: src/logging_setup.py (loguru console plus rotating file sink, idempotent), src/data/lineage.py (LineageRecord dataclass with JSON round-trip and git-commit capture), four pandera schemas under data/schemas/, tests/test_lineage.py, tests/test_schemas.py. Critical: GDP schema bounds set to ge=-25.0, le=25.0 to accommodate the COVID Q2 2020 observation. Verification: 19 out of 19 lineage and schema tests passed.

### Task 3: data downloaders

Files created: src/data/downloaders/{__init__.py, _common.py, ons.py, boe.py, fred.py, boe_yc.py}, tests/test_downloaders.py. OECD as a standalone source was dropped before implementation (PMI and CCI come from FRED). Yahoo was implemented then later removed entirely when the FTSE feature was dropped. During implementation: ONS API migration handled (the retired api.ons.gov.uk v0 API replaced by the website timeseries endpoint with path and dataset segments per series), boe.py IADB URLs verified, boe_yc.py added later as part of the BoE GLC migration. Verification: 22 downloader tests passed (unit plus 3 integration that hit live ONS, FRED, Yahoo at the time of task 3 first pass).

### Task 4: aggregation, features, quality (with cascading config and BoE GLC integration)

Files created: src/data/aggregate.py, src/features/engineer.py, src/data/quality.py, tests/test_aggregate.py, tests/test_engineer.py, tests/test_quality.py. Critical look-ahead leakage tests in place: test_lag_features_are_truly_lagged, test_rolling_uses_only_past_and_present, test_no_future_leakage_across_pipeline. All passed.

gdp_yoy definition agreed: 4-quarter compounded growth from QoQ rates (not a simple delta or pct_change of the rate). Cascaded cpi_yoy removal (redundant since cpi_inflation is already YoY from ONS).

BoE GLC archive integration landed here: src/data/downloaders/boe_yc.py (gilt 2y and 10y both from glcnominalmonthedata.zip, HTTP 200 HTML trap guarded by ZIP signature check, openpyxl parses the "4. spot curve" sheet across all three XLSX files in the archive). gilt_10y_yield moved out of FRED, gilt_2y_yield undeferred, yield_curve_slope wired in. Deferred-feature scaffolding removed in a separate sub-task once the slope was wired: 6 tests deleted (all of which existed solely to assert deferred-skip behaviour), the engineer-dispatcher branch removed, downloader return types tightened from "pd.DataFrame or None" to pd.DataFrame.

Verification: 26 out of 26 task 4 tests passed, critical leakage test passed in isolation, coverage on src/features/engineer.py 95% (at least 80% acceptance).

### Task 5: orchestration, Docker, Makefile

Files created: src/data/build_dataset.py, Dockerfile, docker-compose.yml, .dockerignore. Files modified: Makefile (added download, process, data, all targets), pyproject.toml (added pyarrow at version 14.0 or later).

Pipeline flow: download_all_* runs first; then to_quarterly with the configured aggregation method; then an outer merge on date; then engineer_features; then trim to the window 2000-01-01 through 2025-12-31; then check_final_dataset; then FINAL_DATASET_SCHEMA.validate; then to_parquet. Outer-then-trim is deliberate because engineered features at the 2000 boundary draw on real pre-2000 history (gdp_lag_4 at 2000-Q1 resolves to 0.7, not NaN).

Four bugs surfaced during the first live make data run, none of which the unit tests had caught because they depended on real network responses:

1. trade_balance came through as int64 and failed the RAW_ONS_SCHEMA float64 check. Fix: .astype(float) in the ONS, FRED, BoE parsers.
2. BoE IADB returned 403 Forbidden to the default Python user-agent. Fix: Mozilla user-agent in boe.py, parity with boe_yc.py.
3. NPQT and NMRY are chained-volume GBP million levels, not growth rates. Fix: new qoq_pct_change aggregation method in src/data/aggregate.py, both series re-tagged.
4. business_confidence and consumer_confidence (FRED codes BSCICP02GBM460S and CSCICP02GBM460S) are percentage-balance, not amplitude-adjusted indices. Fix: in_range bounds widened from "85 to 115" to "-60 to 60" in processed_quarterly.py, synthetic frame in test_schemas.py updated to match.

Verification: local make data produced 104 rows by 18 columns, date range 2000-03-31 to 2025-12-31, COVID Q2 2020 gdp_growth at -19.9%, gdp_lag_4 at 2000-Q1 equals 0.7 (real pre-2000 history), yield_curve_slope at 2000-Q1 equals -1.13 (inverted curve, historically correct). docker compose build pipeline succeeded in around 60 seconds (uk-gdp-regime-forecasting:latest). docker compose run --rm pipeline make data succeeded; the container wrote final_dataset.parquet to /app/data/processed/ and the host-mounted volume showed the same shape.

### Task 6: decision log and documentation

Files modified: report/decision-log.md. Files created: data/README.md, src/data/README.md. Decision log: 11 entries total (1 pre-existing "Repository and tooling setup" plus 9 added across task 1 to task 5 plus 1 added for "Feature engineering before date-window trim"). All Source lines stripped on Faith's instruction, intro paragraph updated to match. READMEs: data/README.md documents layout, source and CDID table, refresh procedure, lineage, schemas; src/data/README.md documents module map, the 8-step pipeline flow, aggregation methods, how to add a source, look-ahead discipline, test approach. Verification: grep -c "^## Decision:" report/decision-log.md equals 11, Source count equals 0, both READMEs present.

## Final verification

All required files present (src/data/build_dataset.py, Makefile, Dockerfile, docker-compose.yml, .dockerignore, all schemas, all downloaders, all tests, both READMEs, decision log). Unit tests pass at 83 out of 83 with `pytest -m "not integration"`, 4 deselected (the integration tests). Integration tests pass: live ONS, FRED, BoE GLC downloads all return non-trivial frames; BoE IADB not live-tested in the standard suite (no integration test was written for it) but exercised end-to-end by make data. Coverage at least 80% on src/features/ achieved at 95%. Linting clean: ruff check src/ tests/ clean, black --check src/ tests/ clean. make data succeeds end-to-end locally and in Docker. Final dataset has at least 90 rows (104, exactly the project window). COVID Q2 2020 outlier present (gdp_growth at -19.9%). Docker reproducibility verified: docker compose build pipeline plus docker compose run --rm pipeline make data produced an identical parquet on the host mount.

## Final dataset summary

Path: data/processed/final_dataset.parquet. Shape: 104 rows by 18 columns. Date range: 2000-03-31 to 2025-12-31 (every quarter-end present, sum to 104 which is 26 times 4). Columns in pipeline insertion order: date, gdp_growth, unemployment_rate, cpi_inflation, trade_balance, gfcf_growth, govt_consumption_growth, bank_rate, gbp_usd_rate, brent_oil, business_confidence, consumer_confidence, gdp_lag_1, gdp_lag_4, gdp_rolling_mean_4q, gdp_yoy, business_confidence_rolling_mean_4q, yield_curve_slope. Source mix: ONS 6 (gdp_growth plus 5 predictors), BoE 2 (bank_rate, gbp_usd_rate), FRED 3 (brent_oil, business_confidence, consumer_confidence), BoE_YC 2 (gilt 2y and 10y, intermediate only, dropped after slope is computed).
