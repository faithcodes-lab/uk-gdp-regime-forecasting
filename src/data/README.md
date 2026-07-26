# `src/data/`

The data pipeline. Owns: downloading from four upstream sources, validating raw responses, aggregating to quarter-end frequency, merging, engineering features, running quality + look-ahead checks, validating the final dataset against the processed schema, and writing `data/processed/final_dataset.parquet`.

The companion `src/features/engineer.py` adds engineered features; everything else lives here.

## Module map

```
src/data/
├── config.py              cached YAML loaders for pipeline.yaml / features.yaml / regimes.yaml
├── lineage.py             LineageRecord dataclass + JSON read/write + git_commit capture
├── aggregate.py           to_quarterly(df, method=...) for QE resampling (4 methods)
├── quality.py             check_final_dataset(df): date shape, missing quarters, lag/rolling invariants, COVID-Q2 outlier
├── build_dataset.py       master orchestrator + CLI (--download-only, --process-only, no flag)
└── downloaders/
    ├── _common.py         shared helpers: sentinel check, cache pruning, lineage write
    ├── ons.py             ONS website timeseries JSON (replaced retired api.ons.gov.uk)
    ├── boe.py             BoE IADB CSV exports
    ├── fred.py            FRED /series/observations JSON
    └── boe_yc.py          BoE Government Liability Curve monthly archive (ZIP of XLSX)
```

## Pipeline flow

`src/data/build_dataset.py` orchestrates everything in eight steps:

```
1. download_all_{ons,boe,fred,boe_yc}            raw (date, value) frames in memory + on disk
2. to_quarterly(df, method=cfg.aggregation)      per-series resample to QE
3. merge all quarterly frames on date (outer)    preserves full history pre-2000
4. engineer_features(merged)                     adds 6 engineered features; drops both gilts
5. trim to project date_range                    2000-01-01 → 2025-12-31 (104 quarters)
6. check_final_dataset(df)                       quality + look-ahead invariants
7. FINAL_DATASET_SCHEMA.validate(df)             pandera shape + range checks
8. df.to_parquet(data/processed/final_dataset.parquet)
```

Both step 3 (outer merge) and step 5 (post-engineer trim) are deliberate, outer merge keeps pre-2000 history so engineered features near the project-window start (`gdp_lag_4`, `gdp_yoy`, the 4-quarter rolling means) use real values rather than NaN; the trim then drops the warm-up history so the final dataset is exactly the 104 project quarters.

### Per-source aggregation methods

| Method            | Used by                                       | Behaviour |
|---|---|---|
| `identity`        | `gdp_growth`                                  | Input already quarterly; aligns to QE |
| `quarterly_mean`  | most monthly + daily series                   | Average within each quarter |
| `end_of_period`   | `bank_rate`, `gbp_usd_rate`, gilt yields      | Last value of each quarter (stock variables) |
| `qoq_pct_change`  | `gfcf_growth`, `govt_consumption_growth`      | Derived QoQ % change from a quarterly level series (ONS publishes NPQT and NMRY as CVM SA £m levels; growth is computed here) |

## Adding a new data source

1. Add a `<source>:` block under `data_sources:` in `config/pipeline.yaml` with series codes and per-series aggregation method.
2. Create `src/data/downloaders/<source>.py` with `download_<source>_series(name) -> pd.DataFrame` returning a `(date, value)` frame, validating against the relevant raw schema, and writing a lineage record. Use `cache_raw_response` and `write_raw_csv` from `_common.py` so behaviour stays uniform.
3. Add `download_all_<source>(): dict[str, pd.DataFrame]` to that module.
4. Register both functions in `src/data/downloaders/__init__.py` and add the source to `_DOWNLOADERS` in `build_dataset.py`.
5. Add the series name(s) to `config/features.yaml` (under `raw_predictors_kept` or `intermediate_only_then_dropped`), and add the column(s) to `data/schemas/processed_quarterly.py` with generous range bounds.
6. Add unit tests in `tests/test_downloaders.py` (mock the HTTP layer; assert the sentinel/format/cache/lineage behaviour) and an integration test marked `@pytest.mark.integration` that hits the real endpoint.
7. Update the data-source table in `data/README.md`.

## Look-ahead leakage discipline

Look-ahead leakage is the most serious threat to the validity of this project's results, and the feature pipeline is built specifically to prevent it. Every engineered feature in `src/features/engineer.py` uses one of:

- `Series.shift(n)`:  strictly past
- `Series.rolling(n, min_periods=n)` with default `closed="right"` :past + present only
- Contemporaneous arithmetic (e.g. `gilt_10y − gilt_2y`) : same-time only

The critical tests in `tests/test_engineer.py` are:

- `test_lag_features_are_truly_lagged` - `gdp_lag_k[t]` must equal `gdp_growth[t-k]`
- `test_rolling_uses_only_past_and_present` — mutating `gdp_growth[t+1]` must not change the rolling feature at any `t' ≤ t`
- `test_no_future_leakage_across_pipeline`: poisoning the last row of every raw series must not affect any engineered value at an earlier row

`src/data/quality.py` also re-verifies the lag and rolling invariants against the materialised dataset at the end of the run, so a regression that bypassed the unit tests would still be caught before the parquet is written.

## Testing approach

| Layer | What | Where | Network? |
|---|---|---|---|
| Config loaders | `pipeline_config`, `features_config`, `regimes_config` shape | `tests/test_config.py` | no |
| Schemas | accept valid frames; reject obviously wrong ones | `tests/test_schemas.py` | no |
| Lineage | round-trip, git-commit capture, fallback when git absent | `tests/test_lineage.py` | no |
| Downloaders (unit) | sentinel rejection, parser correctness, end-to-end with mocked HTTP, cache prune, ZIP-signature guard for BoE_YC | `tests/test_downloaders.py` | no |
| Downloaders (integration) | real ONS, FRED, BoE_YC calls, marked `@pytest.mark.integration` | same file | yes (manual) |
| Aggregation | every method round-trips a synthetic frame correctly | `tests/test_aggregate.py` | no |
| Feature engineering | look-ahead invariants, gdp_yoy compounding formula, intermediate drops | `tests/test_engineer.py` | no |
| Quality | every check raises the right error on a deliberately broken synthetic frame | `tests/test_quality.py` | no |

```bash
make test                                    # full unit suite
.venv/bin/pytest -m integration -v --no-cov  # live network integration tests
```

The end-to-end "did the pipeline actually produce a sensible parquet" check is `make data` itself, the four sources, the merge, the engineered features, the date trim, the quality checks, and the schema all have to agree.
